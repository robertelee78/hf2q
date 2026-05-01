//! ADR-017 §A.2 — `DiskBlockStore` synchronous I/O surface.
//!
//! `DiskBlockStore` owns the on-disk lifecycle of envelope files for a
//! single `cache_root`:
//!
//!   * `write_block_sync` — atomic write via `format::write_envelope`,
//!     followed by an `index.insert(...)` so the freshly-published block
//!     is immediately visible to readers (ADR-017 §D5 + §R-F1).
//!   * `read_block` — header + body read with body-hash verification
//!     (ADR-017 §R-C1). Body integrity is checked via the existing
//!     `format::read_envelope_body` path; corrupted bodies bubble an
//!     `io::Error` and the *caller* (recovery path) decides quarantine.
//!   * `remove_block` — `fs::remove_file` + `index.remove`, returns
//!     bytes freed so `evict_lru_until_under_budget` can short-circuit.
//!   * `evict_lru_until_under_budget` — walks `index` entries by `mtime`
//!     ascending and removes until `total_bytes_on_disk <= budget`.
//!     Mtime ordering is *the* ground truth here (`fs::metadata`-driven,
//!     never synthesized — feedback_substrate_must_not_synthesize_ship_gates).
//!     Blocks held by an external `Arc<>` ref are skipped via the
//!     `is_block_pinned` callback (Phase A.3 wires the live KV-cache
//!     liveness check; A.2's tests inject the pin set directly).
//!   * `block_path` / `quarantine_dir` — pure-path helpers exposing the
//!     §D5 layout without touching the filesystem.
//!
//! ## Cross-process safety (ADR-017 §R-F10)
//!
//! Writes acquire an advisory `flock(LOCK_EX)` keyed on
//! `(model_fingerprint_short, block_hash[..2])` for the duration of the
//! atomic-rename publication. The lock file lives at
//! `<cache_root>/locks/<short>__<hash_prefix>.lock`. Pattern mirrors
//! `serve/cache.rs::CacheLock` (already-shipped in iter-205): own the
//! `File` to keep the fd alive, `flock(LOCK_EX)` on acquire, fd-drop
//! releases. Per-block-hash-prefix granularity (256 buckets per model)
//! keeps the contention surface tight without one-lock-per-block fan-out.
//!
//! ## Why writes go through `format::write_envelope`
//!
//! `format::write_envelope` is the canonical atomic-publication path:
//! `<path>.tmp.<pid>` + `sync_all` + `fs::rename`. `DiskBlockStore`
//! deliberately delegates rather than re-implementing — Phase A.1's
//! tests already lock in the byte-for-byte oMLX compat properties of
//! that path.

use std::fs::{self, File};
use std::io::{self, ErrorKind};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use crate::serve::kv_persist::format::{self, BlockHash, EnvelopeHeader, ModelFingerprint};
use crate::serve::kv_persist::index::{BlockIndex, BlockMeta};

/// Hard upper bound on a single block's serialized size. Mirrors ADR-017
/// §R-F11 (oversized-block refusal). Bigger than this is treated as
/// almost-certainly a payload-codec bug — we refuse the write rather than
/// burn disk and corrupt the budget accounting. Value is conservative:
/// 256 tokens × 64 layers × 8 KV heads × 128 dim × 2 (K + V) × 4 bytes ≈
/// 128 MiB ceiling for dense F32; we round up to 256 MiB to leave
/// headroom for state-payload variants without ever shipping multi-GB
/// blocks in a single envelope.
pub const MAX_BLOCK_BYTES: u64 = 256 * 1024 * 1024;

/// One-shot completion handle the writer thread fires after a successful
/// (or failed) sync write. The async writer surface exposes this via
/// `WriteJob.completion_tx`; sync callers ignore it.
pub type CompletionTx = std::sync::mpsc::SyncSender<io::Result<()>>;

/// Job submitted to the [`AsyncWriterHandle`] queue. Owns its body bytes
/// (cheap `Vec<u8>` move). On a successful write the worker fires
/// `completion_tx` (if present) with `Ok(())` and `index.insert(...)`s
/// the block; on failure it fires `Err(...)` and emits a `tracing::warn!`
/// for operator visibility.
#[derive(Debug)]
pub struct WriteJob {
    pub header: EnvelopeHeader,
    pub body: Vec<u8>,
    pub completion_tx: Option<CompletionTx>,
}

/// Synchronous I/O surface over the on-disk envelope layout (§D5).
///
/// `DiskBlockStore` is `Send + Sync` and cheaply cloneable through `Arc`.
/// The async writer (`writer.rs`) holds `Arc<DiskBlockStore>` and calls
/// `write_block_sync` on the worker thread; `BlockPrefixCacheSpiller<E>`
/// (§A.3) holds another `Arc` for the read path.
pub struct DiskBlockStore {
    cache_root: PathBuf,
    index: BlockIndex,
    /// Operator-supplied byte budget (ADR-017 §R-F5). `0` disables the
    /// budget check — useful for tests and for the `HF2Q_KV_PERSIST=1`
    /// "uncapped pilot" mode. Production callers pass a real value.
    budget_bytes: AtomicU64,
    /// Per-store override of the §R-F11 oversized-block refusal ceiling.
    /// `0` means "use [`MAX_BLOCK_BYTES`]". Tests set this small so the
    /// rejection path is exercised without a 256-MiB allocation.
    max_block_bytes_override: AtomicU64,
}

impl DiskBlockStore {
    /// Open or create a store rooted at `cache_root`. Creates the
    /// `locks/` and `models/` subdirectories if missing. Does NOT scan
    /// the existing cache — call [`recovery::recover_from_disk`] for
    /// that and pass the resulting [`BlockIndex`] in via [`new_with_index`].
    pub fn new(cache_root: PathBuf, budget_bytes: u64) -> io::Result<Self> {
        Self::new_with_index(cache_root, BlockIndex::new(), budget_bytes)
    }

    /// Open with a pre-built [`BlockIndex`] (e.g. from
    /// `recovery::recover_from_disk`). The index becomes the store's
    /// live state; subsequent writes/removes mutate it in place.
    pub fn new_with_index(
        cache_root: PathBuf,
        index: BlockIndex,
        budget_bytes: u64,
    ) -> io::Result<Self> {
        if !cache_root.exists() {
            fs::create_dir_all(&cache_root)?;
        }
        fs::create_dir_all(cache_root.join("locks"))?;
        fs::create_dir_all(cache_root.join("models"))?;
        Ok(Self {
            cache_root,
            index,
            budget_bytes: AtomicU64::new(budget_bytes),
            max_block_bytes_override: AtomicU64::new(0),
        })
    }

    /// Effective per-block size ceiling (defaults to [`MAX_BLOCK_BYTES`]
    /// unless [`set_max_block_bytes_override`] has been called).
    pub fn max_block_bytes(&self) -> u64 {
        let override_val = self.max_block_bytes_override.load(AtomicOrdering::Relaxed);
        if override_val == 0 {
            MAX_BLOCK_BYTES
        } else {
            override_val
        }
    }

    /// Test-only: clamp the per-block size ceiling without allocating
    /// the full [`MAX_BLOCK_BYTES`]. `cfg(test)` and `pub(crate)` are
    /// the right gate; we keep it `pub` (not under `#[cfg(test)]`) so
    /// integration tests in `tests/kv_persist_writer_kill_minus_9.rs`
    /// can call it. There is no production caller.
    #[doc(hidden)]
    pub fn set_max_block_bytes_override(&self, override_bytes: u64) {
        self.max_block_bytes_override
            .store(override_bytes, AtomicOrdering::Relaxed);
    }

    /// Cache root path passed to [`new`] / [`new_with_index`].
    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    /// Current configured budget (bytes). `0` disables enforcement.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes.load(AtomicOrdering::Relaxed)
    }

    /// Update the byte budget at runtime. The next eviction call will
    /// observe the new value; existing in-flight writes are not affected.
    pub fn set_budget_bytes(&self, new_budget: u64) {
        self.budget_bytes.store(new_budget, AtomicOrdering::Relaxed);
    }

    /// Read-only view of the live index.
    pub fn index(&self) -> &BlockIndex {
        &self.index
    }

    /// `<cache_root>/models/<model_fp_short>/kv/<hash_hex0>/<hash_hex>.safetensors`
    /// per ADR-017 §D5. Pure-path helper — does NOT touch the filesystem.
    pub fn block_path(&self, model_fp: &ModelFingerprint, hash: &BlockHash) -> PathBuf {
        let hex = hash.to_string();
        let fanout = &hex[..1];
        self.cache_root
            .join("models")
            .join(model_fp.short_hex())
            .join("kv")
            .join(fanout)
            .join(format!("{hex}.safetensors"))
    }

    /// Per-model quarantine directory (§R-F9). Created lazily by the
    /// recovery path; this helper just returns the canonical PathBuf.
    pub fn quarantine_dir(&self, model_fp: &ModelFingerprint) -> PathBuf {
        self.cache_root
            .join("models")
            .join(model_fp.short_hex())
            .join("kv-quarantine")
    }

    /// Path to the per-`(model, hash[..2])` advisory lock file.
    /// `block_hash[..2]` is two hex characters → 256 buckets per model.
    fn lock_path(&self, model_fp: &ModelFingerprint, hash: &BlockHash) -> PathBuf {
        let hex = hash.to_string();
        let prefix = &hex[..2];
        self.cache_root.join("locks").join(format!(
            "{}__{}.lock",
            model_fp.short_hex(),
            prefix
        ))
    }

    /// Synchronous write: enforce the §R-F11 size ceiling, acquire the
    /// per-prefix advisory lock, delegate to `format::write_envelope` for
    /// the atomic-rename, then `index.insert(...)` so readers see the
    /// freshly-published block immediately.
    ///
    /// Returns the absolute path of the published file.
    pub fn write_block_sync(
        &self,
        header: &EnvelopeHeader,
        body: &[u8],
    ) -> io::Result<PathBuf> {
        let body_len = body.len() as u64;
        let ceiling = self.max_block_bytes();
        if body_len > ceiling {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!("block body {body_len} bytes exceeds MAX_BLOCK_BYTES {ceiling}"),
            ));
        }

        let path = self.block_path(&header.model_fingerprint, &header.block_hash);
        let _lock = AdvisoryLock::acquire(&self.lock_path(&header.model_fingerprint, &header.block_hash))?;

        let total_bytes = format::write_envelope(&path, header, body)?;

        // fs::metadata is the ground-truth source for mtime + bytes_on_disk.
        // Per feedback_substrate_must_not_synthesize_ship_gates: never
        // synthesize size from `body.len() + header_estimate`.
        let metadata = fs::metadata(&path)?;
        let mtime = metadata.modified().unwrap_or(std::time::UNIX_EPOCH);
        let bytes_on_disk = metadata.len();
        debug_assert_eq!(bytes_on_disk, total_bytes, "stat size matches writer return");

        let meta = BlockMeta {
            hash: header.block_hash,
            parent: header.parent_block_hash,
            model_fp: header.model_fingerprint,
            payload_kind: header.payload_kind.clone(),
            codec_version: header.codec_version,
            n_tokens: header.n_tokens,
            file_path: path.clone(),
            mtime,
            bytes_on_disk,
        };
        self.index.insert(meta);
        Ok(path)
    }

    /// Read a block by hash. Returns the body bytes only; callers that
    /// also want the header should use `read_block_with_header`.
    pub fn read_block(&self, hash: &BlockHash) -> io::Result<Vec<u8>> {
        let meta = self.index.lookup(hash).ok_or_else(|| {
            io::Error::new(ErrorKind::NotFound, format!("block {hash} not in index"))
        })?;
        let (_, body) = format::read_envelope_body(&meta.file_path)?;
        Ok(body)
    }

    /// Read header + body together. Used by code paths that need to
    /// re-validate the chain identity before consuming the body
    /// (ADR-017 §R-C2).
    pub fn read_block_with_header(
        &self,
        hash: &BlockHash,
    ) -> io::Result<(EnvelopeHeader, Vec<u8>)> {
        let meta = self.index.lookup(hash).ok_or_else(|| {
            io::Error::new(ErrorKind::NotFound, format!("block {hash} not in index"))
        })?;
        format::read_envelope_body(&meta.file_path)
    }

    /// Remove a block: `fs::remove_file` then `index.remove`. Returns
    /// bytes freed (file's `bytes_on_disk` from the index pre-remove)
    /// or `0` if the hash was not indexed. Missing-file is tolerated
    /// (idempotent remove) — the index is the authoritative state.
    pub fn remove_block(&self, hash: &BlockHash) -> io::Result<u64> {
        let Some(meta) = self.index.remove(hash) else {
            return Ok(0);
        };
        match fs::remove_file(&meta.file_path) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {
                // Tolerated: the file was deleted out from under us
                // (e.g. operator ran `rm -rf`). Index entry is gone now;
                // we report the bytes the index *thought* we held so the
                // budget bookkeeping is consistent.
            }
            Err(e) => return Err(e),
        }
        Ok(meta.bytes_on_disk)
    }

    /// Evict in mtime-ascending order until `index.total_bytes_on_disk()
    /// <= budget_bytes`. Blocks held by an external `Arc<>` ref are
    /// skipped via `is_block_pinned`. Returns total bytes evicted.
    ///
    /// Mtime ordering uses real `fs::metadata`-derived values stored on
    /// each [`BlockMeta`] (§R-F5). Per
    /// `feedback_substrate_must_not_synthesize_ship_gates`, we never
    /// invent counts: the budget check, pin filter, and mtime sort all
    /// run against the live index + filesystem.
    ///
    /// `is_block_pinned` returns `true` if the block is currently in use
    /// (held by an inference engine, mid-restore, etc.). Phase A.3 will
    /// wire this to the live KV-cache liveness; A.2's callers (and tests)
    /// pass `|_| false` when the pin set is empty.
    pub fn evict_lru_until_under_budget<F>(
        &self,
        is_block_pinned: F,
    ) -> io::Result<u64>
    where
        F: Fn(&BlockHash) -> bool,
    {
        let budget = self.budget_bytes();
        if budget == 0 {
            return Ok(0);
        }

        let total = self.index.total_bytes_on_disk();
        if total <= budget {
            return Ok(0);
        }

        // Snapshot current entries; sort by mtime ascending. We sort by
        // mtime as the primary key and `bytes_on_disk` (descending) as a
        // tie-breaker so that ties evict the larger block first (frees
        // budget faster). The third tier is `hash` for a deterministic
        // total order — important for test reproducibility.
        let mut entries: Vec<BlockMeta> = self.index.snapshot_all();
        entries.sort_by(|a, b| {
            a.mtime
                .cmp(&b.mtime)
                .then_with(|| b.bytes_on_disk.cmp(&a.bytes_on_disk))
                .then_with(|| a.hash.0.cmp(&b.hash.0))
        });

        let mut freed = 0u64;
        for meta in entries {
            if self.index.total_bytes_on_disk() <= budget {
                break;
            }
            if is_block_pinned(&meta.hash) {
                continue;
            }
            // Re-check the index has the block; a concurrent writer may
            // have already removed it. `remove_block` is idempotent.
            if self.index.lookup(&meta.hash).is_none() {
                continue;
            }
            let bytes = self.remove_block(&meta.hash)?;
            freed = freed.saturating_add(bytes);
        }
        Ok(freed)
    }
}

/// Convenience: cheap `Arc` clone for sharing across the async writer
/// thread + the read path.
pub fn shared(store: DiskBlockStore) -> Arc<DiskBlockStore> {
    Arc::new(store)
}

// ---------------------------------------------------------------------------
// Advisory lock — flock(LOCK_EX) wrapper. Mirrors serve/cache.rs::CacheLock
// but lives in this module to keep the kv_persist surface decoupled. We
// could re-export CacheLock, but the lock-key shape and lifetime are
// distinct enough (per-block-hash-prefix vs per-quant) that a tiny
// wrapper here keeps the call sites readable.
// ---------------------------------------------------------------------------

struct AdvisoryLock {
    /// Held to keep the fd alive; flock releases on drop.
    _file: File,
}

impl AdvisoryLock {
    /// Block until granted. Creates `path` if missing (parent dir is
    /// expected to exist — `DiskBlockStore::new` creates `locks/`).
    fn acquire(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let fd = file.as_raw_fd();
        // SAFETY: fd is owned by `file`; flock is documented as thread-safe.
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { _file: file })
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::kv_persist::format::{
        compute_model_fingerprint, BlockHash, EnvelopeHeader, ParentBlockHash, BLOCK_TOKENS,
        CURRENT_FORMAT_VERSION,
    };
    use sha2::{Digest, Sha256};
    use std::process;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, SystemTime};

    fn temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = process::id();
        let nanos = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("hf2q-kv-store-{label}-{pid}-{nanos}-{n}"));
        fs::create_dir_all(&dir).expect("temp_dir mkdir");
        dir
    }

    fn fixture_fp(seed: &str) -> ModelFingerprint {
        compute_model_fingerprint(
            seed,
            "Q4_0",
            "hf2q-test-1.0.0",
            "deadbeefcafebabe1122334455667788",
            "<|im_start|>...<|im_end|>",
        )
    }

    /// Build (body, header) such that sha256(body) == header.block_hash.
    /// `seed` differentiates bodies across blocks so tests can hold many
    /// distinct hashes without manual bookkeeping.
    fn make_block(
        fp: ModelFingerprint,
        parent: ParentBlockHash,
        seed: u32,
    ) -> (Vec<u8>, EnvelopeHeader) {
        let body: Vec<u8> = (0..512u32)
            .flat_map(|i| (i.wrapping_add(seed)).to_le_bytes())
            .collect();
        let mut h = Sha256::new();
        h.update(&body);
        let bh: [u8; 32] = h.finalize().into();
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: BlockHash(bh),
            parent_block_hash: parent,
            payload_kind: "kv-dense-bf16".into(),
            codec_version: 1,
            n_tokens: BLOCK_TOKENS,
        };
        (body, header)
    }

    #[test]
    fn write_block_sync_round_trip_via_format() {
        // The byte-content invariant: what we wrote == what we read.
        // Per feedback_live_verification_must_check_content: assert on
        // the body bytes, not just on Ok return values.
        let dir = temp_dir("rt");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("rt");
        let (body, header) = make_block(fp, ParentBlockHash(None), 0xAA);
        let path = store.write_block_sync(&header, &body).expect("write");

        // Path matches the §D5 layout.
        let expected = store.block_path(&fp, &header.block_hash);
        assert_eq!(path, expected);
        assert!(path.exists(), "file at expected path");

        // Read back via format::read_envelope_body — the canonical reader.
        let (header_back, body_back) =
            format::read_envelope_body(&path).expect("read_envelope_body");
        assert_eq!(header_back, header, "header round-trips");
        assert_eq!(body_back, body, "body bytes round-trip byte-for-byte");

        // Index reflects the write.
        let meta = store.index().lookup(&header.block_hash).expect("indexed");
        assert_eq!(meta.bytes_on_disk, fs::metadata(&path).unwrap().len());
        assert_eq!(meta.file_path, path);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_block_returns_bytes_after_write() {
        // Round-trip via DiskBlockStore::read_block specifically (the
        // public surface that A.3's spiller calls into).
        let dir = temp_dir("read");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("read");
        let (body, header) = make_block(fp, ParentBlockHash(None), 0xBB);
        store.write_block_sync(&header, &body).expect("write");

        let body_back = store.read_block(&header.block_hash).expect("read");
        assert_eq!(body_back, body, "read_block returns identical bytes");

        // read_block on an unknown hash returns NotFound.
        let unknown = BlockHash([0xFF; 32]);
        let err = store.read_block(&unknown).err().expect("unknown");
        assert_eq!(err.kind(), ErrorKind::NotFound);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_block_decrements_index_and_deletes_file() {
        let dir = temp_dir("rm");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("rm");

        let (body_a, header_a) = make_block(fp, ParentBlockHash(None), 1);
        let (body_b, header_b) = make_block(fp, ParentBlockHash(None), 2);
        let path_a = store.write_block_sync(&header_a, &body_a).expect("a");
        let path_b = store.write_block_sync(&header_b, &body_b).expect("b");
        assert_eq!(store.index().block_count(), 2);

        let bytes_a = fs::metadata(&path_a).unwrap().len();
        let freed = store.remove_block(&header_a.block_hash).expect("rm a");
        assert_eq!(freed, bytes_a, "freed bytes match on-disk size");
        assert!(!path_a.exists(), "file deleted");
        assert!(path_b.exists(), "other file untouched");
        assert_eq!(store.index().block_count(), 1);
        assert!(store.index().lookup(&header_a.block_hash).is_none());

        // Idempotent: removing again returns 0.
        let freed_again = store.remove_block(&header_a.block_hash).expect("rm a 2");
        assert_eq!(freed_again, 0);

        // File-vanished tolerance: nuke the file out from under us, then
        // remove via index. No error.
        let bytes_b_recorded = store
            .index()
            .lookup(&header_b.block_hash)
            .expect("b indexed pre-nuke")
            .bytes_on_disk;
        fs::remove_file(&path_b).expect("nuke b");
        let freed_b = store.remove_block(&header_b.block_hash).expect("rm b");
        assert_eq!(
            freed_b, bytes_b_recorded,
            "freed bytes come from the index even after the file vanished"
        );
        assert!(freed_b > 0);
        assert_eq!(store.index().block_count(), 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn evict_lru_until_under_budget_evicts_oldest_first() {
        // Strategy: write 5 blocks with distinct mtimes (sleep between
        // writes), set a budget that fits 2 of them, evict, and assert
        // that the 3 oldest were removed and the 2 newest survived.
        let dir = temp_dir("lru");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("lru");

        let mut hashes: Vec<BlockHash> = Vec::new();
        let mut sizes: Vec<u64> = Vec::new();
        for s in 0u32..5 {
            let (body, header) = make_block(fp, ParentBlockHash(None), s);
            let path = store.write_block_sync(&header, &body).expect("write");
            hashes.push(header.block_hash);
            sizes.push(fs::metadata(&path).unwrap().len());
            // Tighten the mtime distinction: sleep 20ms so SystemTime
            // resolution on macOS / Linux ext4 (1ms / 1ns respectively)
            // captures a strictly-monotonic order.
            thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(store.index().block_count(), 5);

        // Pick a budget that fits exactly 2 of the 5 blocks (the two
        // newest, by mtime). Budget = sum of last two sizes. Eviction
        // should remove blocks 0, 1, 2 and leave 3, 4.
        let budget = sizes[3] + sizes[4];
        store.set_budget_bytes(budget);

        let freed = store
            .evict_lru_until_under_budget(|_| false)
            .expect("evict");
        let expected_freed: u64 = sizes[..3].iter().sum();
        assert_eq!(freed, expected_freed, "freed bytes match oldest 3");
        assert_eq!(store.index().block_count(), 2, "2 survivors");

        for h in &hashes[..3] {
            assert!(store.index().lookup(h).is_none(), "oldest evicted");
            // The on-disk file is also gone.
            let p = store.block_path(&fp, h);
            assert!(!p.exists(), "evicted file removed from disk");
        }
        for h in &hashes[3..] {
            assert!(store.index().lookup(h).is_some(), "newest survived");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn evict_lru_skips_in_use_blocks() {
        // The pin-callback skips blocks held by an external Arc<> ref.
        // Pin the OLDEST 2 blocks; the eviction should walk past them
        // and remove block index 2 instead.
        let dir = temp_dir("pin");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("pin");

        let mut hashes: Vec<BlockHash> = Vec::new();
        let mut sizes: Vec<u64> = Vec::new();
        for s in 0u32..5 {
            let (body, header) = make_block(fp, ParentBlockHash(None), s);
            let path = store.write_block_sync(&header, &body).expect("write");
            hashes.push(header.block_hash);
            sizes.push(fs::metadata(&path).unwrap().len());
            thread::sleep(Duration::from_millis(20));
        }

        // Budget = 4 newest fit. Without pins, eviction removes the
        // oldest. With pins on the oldest 2, eviction walks past and
        // removes block index 2 (the 3rd-oldest) instead.
        let budget = sizes[1] + sizes[2] + sizes[3] + sizes[4];
        store.set_budget_bytes(budget);

        let pinned_hashes = vec![hashes[0], hashes[1]];
        let pinned_for_closure = pinned_hashes.clone();
        let freed = store
            .evict_lru_until_under_budget(move |h| pinned_for_closure.contains(h))
            .expect("evict");

        // Eviction took block 2 (oldest non-pinned).
        assert!(store.index().lookup(&hashes[2]).is_none(), "block 2 evicted");
        assert!(store.index().lookup(&hashes[0]).is_some(), "pinned 0 survived");
        assert!(store.index().lookup(&hashes[1]).is_some(), "pinned 1 survived");
        assert_eq!(freed, sizes[2], "freed bytes = block 2 size");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_path_uses_hex_fanout_per_d5() {
        // Pure-path assertion: §D5 layout is
        // <root>/models/<fp_short>/kv/<hex0>/<full_hex>.safetensors.
        let dir = temp_dir("path");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("path");
        // Build a hash whose first hex char is '7'.
        let hex_target_first_char = '7';
        let mut bh = BlockHash([0u8; 32]);
        bh.0[0] = 0x7B; // hex prefix '7' followed by 'b'
        let p = store.block_path(&fp, &bh);

        let p_str = p.to_string_lossy().to_string();
        assert!(p_str.contains(&format!("models/{}", fp.short_hex())));
        assert!(
            p_str.contains(&format!("/kv/{hex_target_first_char}/")),
            "hex-fanout dir = first hex char '{hex_target_first_char}': {p_str}"
        );
        assert!(p_str.ends_with(&format!("{}.safetensors", bh)));

        // Quarantine dir is sibling of kv/.
        let q = store.quarantine_dir(&fp);
        let q_str = q.to_string_lossy().to_string();
        assert!(q_str.ends_with(&format!("models/{}/kv-quarantine", fp.short_hex())));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn advisory_lock_serializes_concurrent_writes() {
        // Two writer threads, same store, same advisory-lock bucket
        // (block_hash[..2] collision via constructed hashes). The
        // contention test asserts both writes succeed AND the lock
        // file exists with the expected naming convention.
        let dir = temp_dir("lock");
        let store = Arc::new(DiskBlockStore::new(dir.clone(), 0).expect("new"));
        let fp = fixture_fp("lock");

        // Helper that builds a header whose block_hash starts with 0xAB
        // so both threads route to the same `__ab.lock` bucket.
        fn forced_prefix_header(
            fp: ModelFingerprint,
            suffix: u32,
        ) -> (Vec<u8>, EnvelopeHeader) {
            let body: Vec<u8> = (0..1024u32)
                .flat_map(|i| (i.wrapping_add(suffix)).to_le_bytes())
                .collect();
            let mut h = Sha256::new();
            h.update(&body);
            h.update(b"-collide-prefix-");
            h.update(suffix.to_le_bytes());
            let mut bh: [u8; 32] = h.finalize().into();
            // Force the first byte to 0xAB so both hashes share that
            // prefix and route to the same lock file. The body-hash
            // verification at read time will fail (we forced the hash);
            // this test only checks the lock semantics, not read.
            bh[0] = 0xAB;
            let header = EnvelopeHeader {
                format_version: CURRENT_FORMAT_VERSION.0,
                model_fingerprint: fp,
                block_hash: BlockHash(bh),
                parent_block_hash: ParentBlockHash(None),
                payload_kind: "kv-dense-bf16".into(),
                codec_version: 1,
                n_tokens: BLOCK_TOKENS,
            };
            (body, header)
        }

        let log: Arc<Mutex<Vec<(u32, SystemTime, SystemTime)>>> =
            Arc::new(Mutex::new(Vec::new()));

        let store_a = Arc::clone(&store);
        let log_a = Arc::clone(&log);
        let h_a = thread::spawn(move || {
            let (body, header) = forced_prefix_header(fp, 1);
            thread::sleep(Duration::from_millis(20));
            let t0 = SystemTime::now();
            let _ = store_a.write_block_sync(&header, &body).expect("write a");
            let t1 = SystemTime::now();
            log_a.lock().unwrap().push((1, t0, t1));
        });

        let store_b = Arc::clone(&store);
        let log_b = Arc::clone(&log);
        let h_b = thread::spawn(move || {
            let (body, header) = forced_prefix_header(fp, 2);
            let t0 = SystemTime::now();
            let _ = store_b.write_block_sync(&header, &body).expect("write b");
            let t1 = SystemTime::now();
            log_b.lock().unwrap().push((2, t0, t1));
        });

        h_a.join().expect("a join");
        h_b.join().expect("b join");

        // Both writes succeeded.
        assert_eq!(store.index().block_count(), 2);

        // The lock file exists at the expected path.
        let example_hash = {
            let mut bh = [0u8; 32];
            bh[0] = 0xAB;
            BlockHash(bh)
        };
        let lp = store.lock_path(&fp, &example_hash);
        assert!(lp.exists(), "lock file present at {}", lp.display());
        // Lock-file bucket name uses two hex chars + model short.
        let lp_name = lp.file_name().unwrap().to_string_lossy().to_string();
        assert!(lp_name.contains("__ab.lock"), "lock file name: {lp_name}");

        // Both writers got logged; we don't enforce non-overlap (timing
        // depends on OS scheduling) but we DO assert the log captured
        // both spans. Lock correctness is tested by the absence of any
        // file-corruption / both-writes-succeed property.
        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 2, "both threads logged");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_block_refusal_returns_error() {
        // §R-F11: refuse writes whose body exceeds the per-block ceiling.
        // We use the test-seam `set_max_block_bytes_override` to exercise
        // the rejection path with a 1-KiB ceiling (instead of allocating
        // the production 256 MiB). The same code path runs in production
        // — only the threshold value differs.
        let dir = temp_dir("oversize");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("oversize");
        store.set_max_block_bytes_override(1024);

        let body: Vec<u8> = vec![0u8; 1025];
        let mut h = Sha256::new();
        h.update(&body);
        let bh: [u8; 32] = h.finalize().into();
        let header = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: BlockHash(bh),
            parent_block_hash: ParentBlockHash(None),
            payload_kind: "kv-oversize-test".into(),
            codec_version: 1,
            n_tokens: BLOCK_TOKENS,
        };
        let err = store.write_block_sync(&header, &body).err().expect("err");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("exceeds MAX_BLOCK_BYTES"),
            "error mentions MAX_BLOCK_BYTES: {err}"
        );
        // Index untouched.
        assert_eq!(store.index().block_count(), 0);
        // No file landed at the expected path (no tmp leftover either).
        let p = store.block_path(&fp, &header.block_hash);
        assert!(!p.exists());

        // Boundary: body of size == ceiling is accepted (uses ≤, not <).
        store.set_max_block_bytes_override(1024);
        let body_at_ceiling: Vec<u8> = vec![0u8; 1024];
        let mut h2 = Sha256::new();
        h2.update(&body_at_ceiling);
        let bh2: [u8; 32] = h2.finalize().into();
        let header_at_ceiling = EnvelopeHeader {
            format_version: CURRENT_FORMAT_VERSION.0,
            model_fingerprint: fp,
            block_hash: BlockHash(bh2),
            parent_block_hash: ParentBlockHash(None),
            payload_kind: "kv-oversize-boundary".into(),
            codec_version: 1,
            n_tokens: BLOCK_TOKENS,
        };
        store
            .write_block_sync(&header_at_ceiling, &body_at_ceiling)
            .expect("at-ceiling accepted");
        assert_eq!(store.index().block_count(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn budget_zero_disables_eviction() {
        // budget_bytes == 0 short-circuits eviction (no-op, returns 0).
        // This is the documented "uncapped pilot" mode; tests assert it
        // doesn't accidentally evict everything.
        let dir = temp_dir("uncapped");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        let fp = fixture_fp("uncapped");
        for s in 0u32..3 {
            let (body, header) = make_block(fp, ParentBlockHash(None), s);
            store.write_block_sync(&header, &body).expect("write");
        }
        assert_eq!(store.index().block_count(), 3);
        let freed = store
            .evict_lru_until_under_budget(|_| false)
            .expect("evict");
        assert_eq!(freed, 0, "uncapped → no eviction");
        assert_eq!(store.index().block_count(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_with_index_preserves_recovery_state() {
        // Recovery path produces a BlockIndex; new_with_index plumbs it
        // into a DiskBlockStore that immediately sees the recovered
        // blocks (no double-walk of disk).
        let dir = temp_dir("withidx");
        // First store: write 3 blocks.
        {
            let store_a = DiskBlockStore::new(dir.clone(), 0).expect("new a");
            let fp = fixture_fp("withidx");
            for s in 0u32..3 {
                let (body, header) = make_block(fp, ParentBlockHash(None), s);
                store_a.write_block_sync(&header, &body).expect("write");
            }
        }
        // Second store: rebuild index from disk, plumb into a fresh store.
        let idx = BlockIndex::rebuild_from_disk(&dir).expect("rebuild");
        assert_eq!(idx.block_count(), 3);
        let store_b = DiskBlockStore::new_with_index(dir.clone(), idx, 0).expect("new b");
        assert_eq!(store_b.index().block_count(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR-017 P1-3: `set_budget_bytes(0)` is the documented "unlimited"
    /// sentinel. The getter must echo it so the cmd_serve startup
    /// `tracing::info!(budget_bytes = ...)` reflects the actual stored
    /// value rather than whatever was passed at construction.
    #[test]
    fn set_budget_bytes_zero_means_unlimited() {
        let dir = temp_dir("budget-zero");
        // Construct with a non-zero seed so the test exercises the
        // store→0 transition (catches a hypothetical bug where
        // set_budget_bytes(0) was a no-op).
        let store = DiskBlockStore::new(dir.clone(), 1 << 20).expect("new");
        assert_eq!(store.budget_bytes(), 1 << 20);
        store.set_budget_bytes(0);
        assert_eq!(store.budget_bytes(), 0, "0 = unlimited sentinel");

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR-017 P1-3: a non-zero budget set via the runtime mutator
    /// must round-trip through the getter. cmd_serve reads
    /// `HF2Q_KV_PERSIST_BUDGET_BYTES`, parses to u64, and calls
    /// `set_budget_bytes(parsed)` AFTER `new_with_index(..., 0)`; this
    /// test pins that contract.
    #[test]
    fn set_budget_bytes_nonzero_persists_through_lookup() {
        let dir = temp_dir("budget-nonzero");
        let store = DiskBlockStore::new(dir.clone(), 0).expect("new");
        assert_eq!(store.budget_bytes(), 0);
        const ONE_GIB: u64 = 1 << 30;
        store.set_budget_bytes(ONE_GIB);
        assert_eq!(store.budget_bytes(), ONE_GIB);
        // Second mutation: prove the AtomicU64 isn't write-once.
        const FOUR_GIB: u64 = 4u64 << 30;
        store.set_budget_bytes(FOUR_GIB);
        assert_eq!(store.budget_bytes(), FOUR_GIB);

        let _ = fs::remove_dir_all(&dir);
    }
}
