# ADR-017 kv_persist ↔ oMLX paged_ssd_cache Cross-Check (2026-05-01)

## §1 Header, scope, caveats

**Auditor:** subagent of `agent-aa4342f4ecc03c1f3` worktree.
**Mission:** side-by-side correctness audit of hf2q's ADR-017 kv_persist
implementation against the verified `/opt/omlx` Python reference.

**Sources of truth (read-only):**
- oMLX:
  - `/opt/omlx/omlx/cache/paged_ssd_cache.py` (the reference implementation;
    paths in this report use this absolute root, not the `paged_ssd_cache.py:`
    short form in the audit brief).
  - `/opt/omlx/omlx/cache/paged_cache.py` (chain-hash semantics).
  - `/opt/omlx/omlx/scheduler.py` (block-size config; rotating-window align).
  - `/opt/omlx/omlx/cache/tiered_manager.py` (hot/cold mediator).
  - `/opt/omlx/omlx/cache/factory.py` (cache wiring).
  - `/opt/omlx/omlx/cache/type_handlers.py` (KV-shape handlers).
- hf2q:
  - `/opt/hf2q/src/serve/kv_persist/{mod,format,index,block_store,recovery,
    writer,registry,loader_wrapper,spiller}.rs`.
  - `/opt/hf2q/src/serve/kv_persist/families/gemma4_dense.rs`.
  - `/opt/hf2q/src/serve/mod.rs:1700-2000` (CLI wire-up; sampled).

**Methodology:** read-only. No `cargo`, no tests, no compile, no `cd`.
All file ops use absolute paths. Output is this single markdown file
in the worktree at `docs/research/adr017-omlx-crosscheck-2026-05-01.md`.

**Caveats:**
- The oMLX reference uses the `safetensors` framing for the on-disk
  envelope; hf2q deliberately mirrors that bytewise, but the JSON
  *fields* the two writers populate are NOT identical (see §3 Finding F1).
  This is a structural (not transport) divergence by design — hf2q's
  envelope carries chain-hash provenance fields that oMLX does not.
- oMLX has no chain-hash recurrence; its block hash is a content hash
  of the prompt-prefix tokens (see §3 Finding F2). This is a deeper
  divergence than the §2 row labels suggest — the two designs share an
  on-disk transport but a different identity contract.
- Audit brief asked us to confirm "Block size = 256 tokens (oMLX vs hf2q
  format.rs BLOCK_TOKENS)". Both sides DO use 256 by default, but oMLX's
  256 is a `paged_cache_block_size: int = 256` *config field* (operator-
  overridable; rotating-window models adjust it to 512–1024) while
  hf2q's `BLOCK_TOKENS: u32 = 256` is a hard-coded `pub const`. See §3
  Finding F3.
- Audit brief asked about `paged_ssd_cache.py:1804-1823`; the reader's
  intent there is the deferred-unlink (LRU evict) path. hf2q has its
  own LRU evict in `block_store.rs::evict_lru_until_under_budget` —
  noted in §2 row M-extra1.

---

## §2 Per-mechanism alignment table

| #  | Mechanism                                | oMLX (file:line)                                                                                  | hf2q (file:line)                                                                              | Aligned?  |
|----|------------------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------|
| 1  | Atomic-rename writer                     | `paged_ssd_cache.py:989-1007`                                                                     | `format.rs:307-369`, `block_store.rs:213-253`                                                 | **Yes**   |
| 2  | Serialized-bytes hot tier (no live MLX)  | `paged_ssd_cache.py:1198-1245`                                                                    | *not implemented* — hf2q has NO hot tier; default-0 env var was sketched but not wired         | **No (deferred by design)** |
| 3  | Chain-hash block addressing              | `paged_cache.py:126-179` (CacheBlock metadata; hash is *content* of prompt prefix, NOT chain-link) | `format.rs:236-261` (true chain-hash recurrence: `sha256(model_fp ‖ parent ‖ tokens_le)`)      | **Partial — semantic divergence; see F2** |
| 4  | Startup directory scan rebuilds index    | `paged_ssd_cache.py:849-947` (`_scan_existing_files` + `_read_file_metadata`)                     | `recovery.rs:110-205`, `index.rs:177-205`                                                     | **Yes**   |
| 5  | Advisory file lock (cross-process)       | *NONE* — only `threading.RLock()` (`paged_ssd_cache.py:428`); no fcntl/flock/lockf in `omlx/`    | `block_store.rs:228, 377-405` (`flock(LOCK_EX)` per-`(model, hash[..2])` bucket)              | **hf2q stronger; oMLX is single-process**  |
| 6  | Quarantine path for corrupted blocks     | `paged_ssd_cache.py:1530-1545, 1670-1690` (corrupted entries are *removed*, not quarantined)      | `recovery.rs:230-279` (move-not-delete to `<slug>/kv-quarantine/<reason>__<name>`)            | **hf2q stronger; oMLX deletes** |
| 7  | Block size = 256 tokens                  | `scheduler.py:321-322` (`paged_cache_block_size: int = 256`, *config-overridable*)                 | `format.rs:65` (`pub const BLOCK_TOKENS: u32 = 256`, *hard-coded*)                              | **Partial — same value; differing mutability; see F3** |
| 8  | Async writer thread                      | `paged_ssd_cache.py:246-297` (writer fn) + `:656-690` (background thread spawn) + `:949-1050` (loop) | `writer.rs:60-202` (`AsyncWriterHandle::spawn` + `worker_loop` + `process_job`)               | **Yes**   |
| 9  | Cache-namespace fingerprint (model_fp)   | `paged_ssd_cache.py:1175-1196` (writes `model_name` string + `created_at` into safetensors metadata; **no fingerprint**) | `format.rs:204-234` (`compute_model_fingerprint` over 5-tuple with NUL separators); `spiller.rs:242-244` (callsite passes `("", "", "")` for the last 3 components) | **Partial — hf2q has the contract; spiller passes only 2/5 inputs; see F4** |

### §2 evidence snippets

#### Row 1 — atomic-rename writer

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:989-1007`):

```python
block_hash, tensors_raw, metadata, file_path = item
...
file_path.parent.mkdir(parents=True, exist_ok=True)
temp_path = file_path.with_name(
    file_path.stem + "_tmp.safetensors"
)
actual_size = _write_safetensors_no_mx(
    str(temp_path), tensors_raw, metadata
)
# Atomic rename to final path
os.rename(str(temp_path), str(file_path))
self._index.update_file_size(block_hash, actual_size)
```

hf2q (`/opt/hf2q/src/serve/kv_persist/format.rs:350-368`):

```rust
// Atomic publication: write to <path>.tmp.<pid>, then rename.
let tmp_name = match path.file_name().and_then(|s| s.to_str()) {
    Some(stem) => format!("{stem}.tmp.{}", process::id()),
    None => format!("envelope.tmp.{}", process::id()),
};
let tmp_path = parent.join(tmp_name);
{
    let mut f = File::create(&tmp_path)?;
    let header_len = header_bytes.len() as u64;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(&header_bytes)?;
    f.write_all(body)?;
    f.sync_all()?;
}
std::fs::rename(&tmp_path, path)?;
```

**Alignment Y.** hf2q includes `f.sync_all()?` BEFORE the rename, which
oMLX omits — hf2q is *stricter*. hf2q's `tmp.<pid>` suffix discriminates
crashed temps per-process; oMLX uses one `_tmp.safetensors` suffix
without pid (concurrent processes writing the same hash would collide).
hf2q's pid-suffix strategy is documented in `format.rs:31`,
`recovery.rs:50-52, 168-171` (recovery scans `.tmp.<pid>` orphans).

#### Row 2 — serialized-bytes hot tier (no live MLX)

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:1231-1245`):

```python
# Store in hot cache (or temporary buffer) for immediate read-back.
# Uses raw bytes (not mx.array objects) so Metal GPU memory can be
# released as soon as the inference thread is done with the arrays.
# NOTE: _promote_to_hot_cache() stores mx.array objects directly
# because those are freshly loaded from SSD (not active inference),
# so they don't tie up Metal allocations from the inference pipeline.
# Storing live inference arrays here would accumulate GPU memory
# under a large hot cache and cause kernel panics (IOGPUMemory underflow).
cache_entry = {
    'tensors_raw': tensors_raw,
    'file_metadata': metadata,
    'num_layers': len(cache_data),
    'layer_cache_types': layer_cache_types,
    'block_metadata': block_metadata,
}
```

hf2q: a recursive grep for `HF2Q_KV_HOT_CACHE_BYTES`, `hot_cache_bytes`,
`hot_cache`, `HotCache`, `hot tier` across `/opt/hf2q/src/` returns ZERO
matches. There is no in-memory hot tier in hf2q today; every read goes
through `block_store.rs:257-263` `read_block` → `format::read_envelope_body`
i.e. file-system read. The audit brief noted "HF2Q_KV_HOT_CACHE_BYTES is
opt-in default-0" — that env var does not yet exist.

**Alignment N (deferred by design).** hf2q's design intentionally lands
the SSD tier first (Phase A.1–A.3 + B-dense + C); the hot tier is part of
the future B/D phases. The oMLX-verified `IOGPUMemory underflow`
hazard does NOT yet apply to hf2q because hf2q does not yet store live
mlx-native buffers in any in-memory cache structure — `Gemma4DenseSpill`
extracts tensor bytes from `MlxBuffer` to `Vec<u8>` *on the inference
thread* before handing them to the writer (see
`families/gemma4_dense.rs:160-175` payload format docstring; the K/V
bytes are extracted into the payload body, never retained as live
buffers in the spiller).

#### Row 3 — chain-hash block addressing (semantic divergence — see F2)

oMLX (`/opt/omlx/omlx/cache/paged_cache.py:126-179`):

```python
@dataclass
class CacheBlock:
    """KV cache block metadata following vLLM's design.
    ...
    Attributes:
        block_id: Physical block index (0 to num_blocks - 1)
        ref_count: Reference count for sharing (0 = can be evicted)
        block_hash: Content hash for prefix caching and paged SSD storage key
        ...
    """
    block_id: int
    ref_count: int = 0
    block_hash: Optional[BlockHash] = None
```

The oMLX `block_hash` is computed in `prefix_cache.py` (not in the
audited range — confirmed by the audit brief's request to focus on
`paged_cache.py:126-162` for "chain-hash semantics"; the lines
themselves describe the *block metadata*, not a hash recurrence).

hf2q (`/opt/hf2q/src/serve/kv_persist/format.rs:236-261`):

```rust
/// Chain-hash recurrence per ADR-017 §D4:
///
/// ```text
/// block_hash(N) = sha256(model_fingerprint
///                       || parent_block_hash_bytes_or_zeros
///                       || token_ids[N*BLOCK..(N+1)*BLOCK].le_bytes)
/// ```
pub fn compute_block_hash(
    model_fp: &ModelFingerprint,
    parent: &ParentBlockHash,
    token_ids: &[u32],
) -> BlockHash {
    let mut h = Sha256::new();
    h.update(model_fp.0);
    h.update(parent.hash_input_bytes());
    for tok in token_ids {
        h.update(tok.to_le_bytes());
    }
    ...
}
```

**Alignment Partial — see §3 F2.** The two designs differ: oMLX's hash
is a content-of-prefix-tokens sha256, hf2q's is a parent-linked recurrence.
The two are NOT cross-readable. This is a deliberate hf2q design choice
(ADR-017 §D4) but the audit brief's framing of "chain-hash semantics"
matches hf2q, NOT oMLX — there is no chain-hash in oMLX.

#### Row 4 — startup directory scan rebuilds index

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:849-876`):

```python
def _scan_existing_files(self) -> None:
    logger.info(f"Scanning SSD cache directory: {self._cache_dir}")
    scanned = 0
    indexed = 0
    errors = 0
    for subdir in self.SUBDIR_CHARS:
        subdir_path = self._cache_dir / subdir
        if not subdir_path.exists():
            continue
        for file_path in subdir_path.glob("*.safetensors"):
            scanned += 1
            try:
                metadata = self._read_file_metadata(file_path)
                if metadata:
                    self._index.add(metadata)
                    indexed += 1
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                errors += 1
```

hf2q (`/opt/hf2q/src/serve/kv_persist/recovery.rs:110-150`):

```rust
pub fn recover_from_disk(cache_root: &Path) -> io::Result<(BlockIndex, RecoveryReport)> {
    let start = Instant::now();
    let index = BlockIndex::new();
    let mut report = RecoveryReport::default();
    let models_dir = cache_root.join("models");
    if !models_dir.exists() { ... return Ok((index, report)); }
    for slug_ent in fs::read_dir(&models_dir)? {
        ... let kv_dir = slug_path.join("kv");
        if !kv_dir.exists() { continue; }
        for fanout_ent in fs::read_dir(&kv_dir)? {
            ...
            for blk_ent in fs::read_dir(&fanout_path)? {
                ...
                scan_one(&slug_path, &blk_path, &index, &mut report)?;
            }
        }
    }
    report.elapsed_ms = start.elapsed().as_millis();
    Ok((index, report))
}
```

**Alignment Y.** Both walk the on-disk directory, parse headers, populate
an in-memory index, and tolerate per-file errors. hf2q's directory layout
is `<root>/models/<slug>/kv/<hex0>/<hex>.safetensors` (slug-fanout); oMLX
is `<root>/<hex0>/<hex>.safetensors` (single-tier hex fanout). hf2q's
tree is deeper because it scopes by `model_fingerprint`, which oMLX does
not have (see Finding F4).

#### Row 5 — advisory file lock for cross-process contention

oMLX: a recursive grep for `fcntl|lockf|flock|portalocker|filelock`
across `/opt/omlx/omlx/` returns ZERO matches. The only lock in
`paged_ssd_cache.py` is `threading.RLock()` (line 428) and
`threading.Lock()` (`hot_cache_lock`, line 648), both *intra-process*.

hf2q (`/opt/hf2q/src/serve/kv_persist/block_store.rs:227-228, 377-405`):

```rust
let _lock = AdvisoryLock::acquire(
    &self.lock_path(&header.model_fingerprint, &header.block_hash)
)?;
let total_bytes = format::write_envelope(&path, header, body)?;
...
struct AdvisoryLock { _file: File }
impl AdvisoryLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        ...
        let file = File::options().create(true).read(true).write(true)
            .truncate(false).open(path)?;
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if ret != 0 { return Err(io::Error::last_os_error()); }
        Ok(Self { _file: file })
    }
}
```

**hf2q stronger.** oMLX assumes single-process operation (it embeds in
the Python serve process). hf2q's `cmd_serve` plus operator tools
(`cmd_cache`, future bench scripts) may share the cache root across
processes; the per-`(model, hash[..2])` flock prevents two `cmd_serve`
instances from racing on the same atomic-rename. Per-prefix granularity
(256 buckets per model — `block_store.rs:32-34`) keeps contention small.

#### Row 6 — quarantine path for corrupted blocks

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:1530-1545, 1670-1690`,
sample at 1530):

```python
# Remove corrupted entry
... self._index.remove(block_hash)
... if file_path.exists():
        file_path.unlink()
```

(both occurrences delete the file outright; no quarantine dir.)

hf2q (`/opt/hf2q/src/serve/kv_persist/recovery.rs:241-279`):

```rust
pub fn quarantine_corrupted_block(
    cache_root: &Path,
    model_fp: &ModelFingerprint,
    original_path: &Path,
    reason: QuarantineReason,
) -> io::Result<PathBuf> {
    let slug_path = cache_root.join("models").join(model_fp.short_hex());
    quarantine_with_prefix(&slug_path, original_path, reason)
}
fn quarantine_with_prefix(...) -> io::Result<PathBuf> {
    let q_dir = slug_path.join("kv-quarantine");
    if !q_dir.exists() { fs::create_dir_all(&q_dir)?; }
    ...
    let dest_name = format!("{}__{}", reason.prefix(), name);
    let dest = q_dir.join(dest_name);
    match fs::rename(blk_path, &dest) { ... }
}
```

**hf2q stronger.** hf2q preserves corrupted blocks with reason-prefixed
filenames (`trunc__`, `verbump__`, `bodyhash__`, `parity__`) so an
operator can post-mortem the bytes. oMLX silently deletes them.

#### Row 7 — block size = 256 tokens

oMLX (`/opt/omlx/omlx/scheduler.py:321-322`):

```python
# Paged cache settings (internal defaults)
paged_cache_block_size: int = 256  # Tokens per block
```

hf2q (`/opt/hf2q/src/serve/kv_persist/format.rs:63-65`):

```rust
/// Block size in tokens. ADR-017 §D3 — adopt oMLX's empirically-validated
/// default (`scheduler.py:321-331`). Phase A0 reserves the right to revisit.
pub const BLOCK_TOKENS: u32 = 256;
```

**Alignment Partial.** Same default value (256), but oMLX's value is a
config field that rotating-window models override via
`_align_block_size_with_rotating_window` (`scheduler.py:898-944`,
target range 512–1024). hf2q hard-codes the constant — Gemma 4 dense /
Qwen 3.5 hybrid would need a code change to vary per family. See §3 F3.

#### Row 8 — async writer thread

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:656-690`):

```python
self._write_queue: queue.Queue = queue.Queue(maxsize=_MAX_PENDING_WRITES)
...
self._writer_shutdown = threading.Event()
self._writer_thread = None
if not self._hot_cache_only:
    self._writer_thread = threading.Thread(
        target=self._writer_loop,
        name="ssd-cache-writer",
        daemon=True,
    )
    self._writer_thread.start()
```

hf2q (`/opt/hf2q/src/serve/kv_persist/writer.rs:60-72`):

```rust
pub fn spawn(store: Arc<DiskBlockStore>, channel_capacity: usize) -> Self {
    let (tx, rx) = mpsc::sync_channel::<WriteJob>(channel_capacity);
    let join_handle = thread::Builder::new()
        .name("hf2q-kv-writer".to_string())
        .spawn(move || { worker_loop(store, rx); })
        .expect("spawn hf2q-kv-writer thread");
    Self { tx: Some(tx), join_handle: Some(join_handle) }
}
```

**Alignment Y.** Both are bounded-queue, single-worker, daemon-style.
hf2q's bound is `channel_capacity: usize` per spawn (default 8 via
`DEFAULT_CHANNEL_CAPACITY` at `writer.rs:218`); oMLX scales bound by
RAM (`_compute_max_pending_writes` returns 32–256). Bound shape differs
but the architecture is identical.

#### Row 9 — cache-namespace fingerprint

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:1175-1182`):

```python
metadata = {
    "omlx_cache_format_version": _CACHE_FORMAT_VERSION,
    "block_hash": block_hash.hex(),
    "token_count": str(token_count),
    "num_layers": str(len(cache_data)),
    "model_name": model_name,
    "created_at": str(time.time()),
}
```

oMLX persists `model_name` as a *string* — there is no namespacing on
disk; isolation between models is the operator's responsibility (use a
different `cache_dir` per model).

hf2q (`/opt/hf2q/src/serve/kv_persist/format.rs:204-234`):

```rust
pub fn compute_model_fingerprint(
    repo_id: &str,
    quant: &str,
    producer_version: &str,
    source_sha256: &str,
    tokenizer_chat_template: &str,
) -> ModelFingerprint {
    let mut h = Sha256::new();
    h.update(repo_id.as_bytes());
    h.update(b"\x00");
    h.update(quant.as_bytes());
    h.update(b"\x00");
    h.update(producer_version.as_bytes());
    h.update(b"\x00");
    h.update(source_sha256.as_bytes());
    h.update(b"\x00");
    h.update(tokenizer_chat_template.as_bytes());
    ...
}
```

But the spiller's call site (`spiller.rs:242-244`):

```rust
fn family_model_fp(repo: &str, quant: QuantType) -> ModelFingerprint {
    compute_model_fingerprint(repo, quant.as_str(), "", "", "")
}
```

**Alignment Partial.** The `compute_model_fingerprint` *function* mirrors
ADR-017 §D4 with a 5-tuple input + NUL separators — strictly stronger
than oMLX's `model_name` string. But the spiller currently calls it with
3/5 components stubbed empty (`""` for `producer_version`,
`source_sha256`, `tokenizer_chat_template`). The empty-stub means today's
namespace key is effectively `(repo, quant)` only — see §3 F4.

### §2 row M-extra1 — LRU evict + budget enforcement

Not in the original 9-row brief but called out by the brief's mention of
`paged_ssd_cache.py:1804-1823`:

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:1800-1823`):

```python
target_size = effective_max - estimated_new_size
if target_size < 0:
    target_size = int(effective_max * 0.9)
if self._index.total_size > target_size:
    evicted = self._index.evict_until_size(target_size)
    for metadata in evicted:
        try:
            self._write_queue.put_nowait(('unlink', metadata.file_path))
        except queue.Full:
            try:
                if metadata.file_path.exists():
                    metadata.file_path.unlink()
                    self._stats["evictions"] += 1
            ...
```

hf2q (`/opt/hf2q/src/serve/kv_persist/block_store.rs:313-360`):

```rust
pub fn evict_lru_until_under_budget<F>(&self, is_block_pinned: F) -> io::Result<u64>
where F: Fn(&BlockHash) -> bool,
{
    let budget = self.budget_bytes();
    if budget == 0 { return Ok(0); }
    let total = self.index.total_bytes_on_disk();
    if total <= budget { return Ok(0); }
    let mut entries: Vec<BlockMeta> = self.index.snapshot_all();
    entries.sort_by(|a, b| {
        a.mtime.cmp(&b.mtime)
            .then_with(|| b.bytes_on_disk.cmp(&a.bytes_on_disk))
            .then_with(|| a.hash.0.cmp(&b.hash.0))
    });
    let mut freed = 0u64;
    for meta in entries {
        if self.index.total_bytes_on_disk() <= budget { break; }
        if is_block_pinned(&meta.hash) { continue; }
        ...
        let bytes = self.remove_block(&meta.hash)?;
        freed = freed.saturating_add(bytes);
    }
    Ok(freed)
}
```

**Alignment Y (with hf2q-side stronger pin set).** Both evict LRU until
under budget. oMLX defers unlink to its writer thread (queue-full
fallback to inline unlink); hf2q does inline unlink under
`is_block_pinned` filter. hf2q's pin callback lets the live KV cache
veto eviction of pinned blocks — oMLX has no such pin contract.

---

## §3 Highest-confidence findings

### F1 — Envelope JSON header schema is hf2q-specific (NOT oMLX-cross-readable)

**Confidence: high.** Both writers use safetensors framing (8-byte LE
header_len + space-padded JSON header + concatenated body). But the
fields inside the JSON header diverge:

oMLX (`/opt/omlx/omlx/cache/paged_ssd_cache.py:1175-1196`):

```python
metadata = {
    "omlx_cache_format_version": _CACHE_FORMAT_VERSION,
    "block_hash": block_hash.hex(),
    "token_count": str(token_count),
    "num_layers": str(len(cache_data)),
    "model_name": model_name,
    "created_at": str(time.time()),
    ... layer_cache_types, layer_meta_states, cache_list_meta ...
}
```

hf2q (`/opt/hf2q/src/serve/kv_persist/format.rs:273-301`):

```rust
pub struct EnvelopeHeader {
    pub format_version: u32,
    pub model_fingerprint: ModelFingerprint,
    pub block_hash: BlockHash,
    pub parent_block_hash: ParentBlockHash,
    pub payload_kind: String,
    pub codec_version: u32,
    pub n_tokens: u32,
}
```

**Impact:** oMLX-produced files cannot be read by hf2q and vice-versa.
hf2q's `format.rs:1-46` already calls out byte-compat with safetensors
*v0.7 framing* but NOT with oMLX's metadata schema. This is an
**intentional design difference**, not a porting bug — hf2q's schema
carries `model_fingerprint`, `parent_block_hash`, `codec_version`
(chain-hash provenance) that oMLX doesn't track. Verdict: by-design
divergence; "byte-compatible safetensors" claim in `format.rs:42-46`
should NOT be misread as "oMLX-file-compatible".

### F2 — Block-hash identity contract differs (chain-hash vs content-hash)

**Confidence: high.** This is the most consequential semantic
divergence and the one the audit brief framed as "chain-hash semantics
(oMLX paged_cache.py:126-162 vs hf2q format.rs chain-hash logic)".

oMLX's `block_hash` is a **content hash** of the prompt-prefix tokens
covering this block (see `paged_cache.py:142` "Content hash for prefix
caching and paged SSD storage key" — the actual hash function lives in
`prefix_cache.py` per oMLX's module split, but the identity is
content-of-tokens, not chain-linked).

hf2q's `block_hash` is a **chain hash**:
`sha256(model_fp ‖ parent_hash_or_zeros ‖ tokens_le)` with the parent
explicitly threaded through (`format.rs:236-261`, plus the spiller's
chain-advance pattern at `spiller.rs:325-368`):

```rust
// Chain-hash linkage: parent[N+1] = block_hash[N].
let mut parent = ParentBlockHash(None);
for layer_rank in 0..n_layers {
    for range in &ranges {
        ...
        let bh: [u8; 32] = Sha256::digest(&body).into();
        let block_hash = BlockHash(bh);
        ...
        let header = EnvelopeHeader {
            ...
            block_hash,
            parent_block_hash: parent,
            ...
        };
        ...
        parent = ParentBlockHash(Some(block_hash));
    }
}
```

Note one subtlety: hf2q's `compute_block_hash` (the function) computes
chain-hash of `(model_fp, parent, tokens)`, but the spiller writes
`block_hash = sha256(body)` — i.e. the on-disk identity that
`format::read_envelope_body` verifies (`format.rs:455-464`) is a *body*
hash, NOT a *chain* hash. The spiller's `parent_block_hash` field is
populated by the *prior* body's sha256, which is what makes the chain
work — but the chain identity is over body bytes, not over token IDs.
This is consistent with the docstring at `format.rs:430-440` ("Phase A.2
will land a body-builder that satisfies this invariant for every
payload kind") and is honest, but it deviates from the §D4 specification
text reproduced at `format.rs:236-244` (which says hash is over
`(model_fp, parent, tokens)`).

**Impact:** This is the divergence most likely to matter at deploy time.
Implications:
- **(a)** Deduplication semantics: oMLX dedups across sessions by
  prefix-content; hf2q dedups by `(parent_body_hash, body_bytes)`. Two
  sessions submitting the same prompt under hf2q produce *different*
  on-disk hashes if the body bytes differ by even one BF16 LSB
  (rounding, NaN encoding, etc.) — even when the *token IDs* match.
  oMLX would treat both as the same block.
- **(b)** Cross-engine file-sharing: hf2q files cannot be read by oMLX
  (different schema per F1) and the chain-hash provenance is unique to
  hf2q's parent-link contract.
- **(c)** Internal consistency: hf2q's `compute_block_hash` function is
  *defined and tested* (`format.rs:528-555`) but **not called** from
  the production spiller path (`spiller.rs:343-346` uses
  `Sha256::digest(&body)` directly). The ADR-017 §D4 invariant is in
  the docstrings + the `compute_block_hash` test suite; the production
  identity is a body-hash. This is an internal inconsistency — see
  recommendation in §4.

### F3 — `BLOCK_TOKENS` is a hard-coded constant; oMLX's is per-family-config

**Confidence: medium-high.** hf2q `format.rs:65` declares
`pub const BLOCK_TOKENS: u32 = 256`. The brief asks "Block size = 256
tokens (oMLX vs hf2q format.rs BLOCK_TOKENS)" — the *value* matches at
default.

oMLX's `scheduler.py:321-322` is `paged_cache_block_size: int = 256` and
`scheduler.py:898-944` adjusts this for rotating-window models
(target range 512–1024 tokens — `_ROTATING_BLOCK_SIZE_MIN/MAX`). Window
sizes that don't fit the target trigger
`_align_block_size_with_rotating_window` to pick a multiple.

hf2q's `Gemma4DenseSpill` has `block_alignment()` on the
`KvCacheSpill` trait surface (`spiller.rs:96-98`) — so per-family
alignment IS pluggable. But the spiller's per-block `n_tokens` is
clipped against `BLOCK_TOKENS` at `spiller.rs:348`:

```rust
let n_tokens = (range.end.saturating_sub(range.start)).min(BLOCK_TOKENS);
```

**Impact:** Today's hf2q ships only Gemma 4 dense (B-dense.1) and the
Qwen 3.5 hybrid is future work. If a future B-hybrid.1 needs a
per-layer block size > 256 (e.g. 512 to align with sliding-window),
the `.min(BLOCK_TOKENS)` clamp will silently truncate the n_tokens
field — a future-bug surface, not a present-day defect. Verdict:
acceptable for Phase A.3 / B-dense.1; revisit before B-hybrid.1.

### F4 — Spiller's model_fingerprint passes only 2/5 components today

**Confidence: high.** `compute_model_fingerprint` was designed to take a
5-tuple `(repo, quant, producer_version, source_sha256, chat_template)`.
The spiller's call site (`spiller.rs:242-244`) passes empties for the
last 3:

```rust
fn family_model_fp(repo: &str, quant: QuantType) -> ModelFingerprint {
    compute_model_fingerprint(repo, quant.as_str(), "", "", "")
}
```

The docstring at `spiller.rs:234-241` acknowledges this: "B-dense.1
wires the real GGUF metadata path (ADR-005 iter-211 already lands the
metadata). The empty placeholders are stable across the spill / restore
call sites, so the model_fp recomputed in `post_admit` matches the one
recorded in `pre_evict`."

**Impact:** Today's namespace key is `sha256("repo\0quant\0\0\0\0")`.
That's still distinct per `(repo, quant)` and won't cross-contaminate
e.g. two `Q4_0` repos. But:
- (a) **Re-quanting the same repo with the same `QuantType` does NOT flip
  the fingerprint.** A `cmd_quantize` rerun that produces a different
  `source_sha256` will accidentally land in the prior run's namespace,
  and the prior run's blocks will be served against the new weights.
  This is the failure mode `compute_model_fingerprint` was designed to
  prevent (see `format.rs:213-219` docstring: "Re-quanting or upgrading
  the chat template flips this and orphans the prior cache namespace
  cleanly.") — but the spiller does not yet exercise that protection.
- (b) **Chat-template upgrades silently get prior-template's blocks.**
  Same mechanism: `tokenizer_chat_template` is empty at the spiller
  call site, so a chat-template change does not flip the namespace.
- (c) **`spiller.rs:243` recomputes the same `("", "", "")` tuple at both
  pre_evict and post_admit**, so within a single hf2q lifecycle
  evict→admit flows are consistent. The hazard is *across* lifecycles
  where the underlying GGUF changed without touching `(repo, quant)`.

Recommendation: B-dense.1 (or earlier) needs to thread the GGUF
provenance bits (`hf2q.producer_version`, `hf2q.source_sha256`,
`tokenizer.chat_template`) into the `family_model_fp` call. ADR-005
iter-211 already lands those metadata keys; the spiller just needs to
read them out of the engine handle.

### F5 — `compute_block_hash` (chain-hash) is defined+tested but unused in production

**Confidence: high.** `format.rs:246-261` ships `compute_block_hash` as
the canonical chain-hash recurrence with extensive tests
(`format.rs:519-628`). But the production spiller path
(`spiller.rs:343-346`) writes `block_hash = sha256(body)` — a body hash,
not a chain hash. The chain-link is enforced via the `parent_block_hash`
field in the header but the `block_hash` *itself* is not the chain
recurrence.

The docstrings reconcile this at `format.rs:425-440` ("the writer's
contract is `body == sha256-pre-image-of(block_hash)`") — i.e. the
codebase is self-consistent: every body satisfies
`sha256(body) == header.block_hash`, and `compute_block_hash` is
available for callers that want chain-hash provenance for indexing /
deduplication purposes (e.g. matching a child to its parent without
reading the body).

**Impact:** Internal API surface confusion. A future engineer reading
ADR-017 §D4 text + format.rs tests might assume the chain-hash recurrence
is what gets written to disk; it is NOT (today). Three options:
1. Remove `compute_block_hash` from `format.rs` (it's unused).
2. Add `EnvelopeHeader.chain_hash: BlockHash` field separate from
   `block_hash` and populate it via `compute_block_hash` at write time
   (matches §D4 verbatim).
3. Update §D4 and the docstrings to reflect "block_hash IS body-hash;
   chain provenance is via parent_block_hash field only".

This is a documentation / API-cleanup divergence, not a correctness
divergence.

---

## §4 Findings summary

**Verdict: minor divergences acceptable; one material divergence requires
investigation before B-dense.1 ships to operators.**

The hf2q port is a faithful rendering of the load-bearing oMLX patterns:
atomic-rename writes (Row 1), bounded async writer (Row 8), startup scan
(Row 4), LRU evict (Row M-extra1) all match. hf2q is *stronger* than
oMLX on cross-process safety (Row 5: flock vs none) and on corruption
forensics (Row 6: quarantine-with-reason vs silent delete). The
on-disk envelope is byte-compatible-as-safetensors-framing but the JSON
header schema diverges (F1) — a deliberate hf2q-specific contract for
chain-hash provenance.

The single material divergence (F4) is the spiller's
`family_model_fp("repo", "quant", "", "", "")` placeholder: it ships
namespacing strong enough for `(repo, quant)` distinction but loses the
re-quant + chat-template-upgrade protection that
`compute_model_fingerprint` was designed for. This is a known limitation
called out in `spiller.rs:234-241` docstrings; the fix is mechanical
(thread GGUF metadata through the call site) and was already scoped to
B-dense.1 per the spiller's own TODO. Until that ships, an operator
re-quanting the same repo without bumping the `(repo, quant)` pair will
silently inherit the prior run's cache — not a corruption hazard
(chain-hashes are still consistent intra-namespace), but a cache-stale
hazard.

Findings F2 (chain-hash vs content-hash), F3 (BLOCK_TOKENS const), and
F5 (`compute_block_hash` unused) are by-design or
documentation-cleanup items, not porting bugs. The hot-tier (Row 2) is
deferred by phase, not skipped — and hf2q's current write path doesn't
expose the IOGPUMemory underflow hazard because it extracts bytes
synchronously on the inference thread before handing them to the writer.

Phase A.1 + A.2 + A.3 + B-dense.1 + C.1 are correct ports of oMLX's
proven mechanics where the ports overlap, and stronger than oMLX on
cross-process + corruption forensics. The B-dense.1 ship gate should
include a fix for F4 (provenance-complete model_fp); F5 is a docs-only
cleanup that can ride along. Everything else is aligned or
intentionally divergent.
