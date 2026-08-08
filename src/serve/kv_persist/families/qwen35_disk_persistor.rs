//! ADR-027 Phase A iter-5 — disk persistence for the Qwen3.5/3.6 hybrid
//! KV-cache snapshot via the QH35 envelope.
//!
//! Provides the cold-process resume path: serialize a
//! `HybridKvCacheSnapshot` through `qwen35_hybrid_persistor::serialize_*`
//! and write it to `~/.cache/hf2q/qwen35_kv/<fingerprint>/<lcp_key>.bin`;
//! on next process start, hydrate the in-memory `LcpRegistry` by walking
//! the directory and deserializing each file.
//!
//! This module is the disk back-end ONLY — wiring into `Qwen35LoadedModel`
//! and the `cmd_serve` / `cmd_generate_qwen35` env-toggle landed in
//! iter-6 (gated separately because that touches the engine load path).
//!
//! # File-on-disk contract
//!
//! ```text
//! <cache_dir>/<fingerprint>/<lcp_key_hex>.bin
//! ```
//!
//! `<fingerprint>` is a 16-hex-char (8-byte) digest of the
//! `Qwen35HybridConfig` compatibility fields. Request-only sequence
//! capacity is intentionally excluded: QH35 v5 compact checkpoints carry
//! their own prefix length and can hydrate into any larger live cache.
//! `<lcp_key_hex>`
//! is the lcp_key's stable hex form (mirrors what `LcpRegistry` would
//! produce for the same key in memory).
//!
//! Atomic writes use the standard tempfile-and-rename pattern so a
//! crash mid-write leaves the persisted file intact.
//!
//! # Iter sequence
//!
//! - **Iter 5 (this commit)**: write / read / hydrate_registry surface
//!   + 4 unit tests (round-trip via tempdir; no GPU model required).
//! - **Iter 6**: thread `Qwen35DiskPersistor` into `Qwen35LoadedModel`
//!   under `HF2Q_KV_PERSIST=<dir>` and validate cold-process LCP resume
//!   on Qwen 3.6 35B-A3B-APEX-Q5_K_M.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::MlxDevice;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::inference::models::qwen35::kv_cache::HybridKvCacheSnapshot;

use super::qwen35_hybrid_persistor::{
    deserialize_hybrid_with_sidecar, serialize_hybrid_with_sidecar, FullAttnCodec,
    LcpSidecarMetadata, Qwen35HybridConfig,
};

/// Qwen checkpoints are large enough that a conventional bounded channel can
/// retain several gigabytes during a cold chunked prefill. This single pending
/// slot coalesces queued writes instead: one job may be in flight and the slot
/// always holds the newest checkpoint submitted while that write runs.
const QWEN35_ASYNC_PENDING_SLOTS: usize = 1;
const QWEN35_ASYNC_IDLE_GRACE: Duration = Duration::from_secs(2);

struct Qwen35DiskWriteJob {
    cfg: Qwen35HybridConfig,
    lcp_key_hex: String,
    snapshot: Arc<HybridKvCacheSnapshot>,
    sidecar: LcpSidecarMetadata,
}

struct Qwen35AsyncWriterState {
    pending: Option<Qwen35DiskWriteJob>,
    in_flight: bool,
    shutdown: bool,
    active_requests: usize,
    idle_since: Option<Instant>,
}

impl Default for Qwen35AsyncWriterState {
    fn default() -> Self {
        Self {
            pending: None,
            in_flight: false,
            shutdown: false,
            active_requests: 0,
            idle_since: Some(Instant::now()),
        }
    }
}

struct Qwen35AsyncWriter {
    shared: Arc<(Mutex<Qwen35AsyncWriterState>, Condvar)>,
    join_handle: Option<JoinHandle<()>>,
}

impl Qwen35AsyncWriter {
    fn spawn(cache_dir: PathBuf, budget_bytes: u64) -> Result<Self> {
        let shared = Arc::new((
            Mutex::new(Qwen35AsyncWriterState::default()),
            Condvar::new(),
        ));
        let worker_shared = Arc::clone(&shared);
        let join_handle = thread::Builder::new()
            .name("hf2q-qwen35-kv-writer".to_string())
            .spawn(move || {
                let sync_target = Qwen35DiskPersistor {
                    cache_dir,
                    budget_bytes,
                    async_writer: None,
                };
                qwen35_async_writer_loop(&sync_target, &worker_shared);
            })
            .context("spawn hf2q-qwen35-kv-writer thread")?;
        Ok(Self {
            shared,
            join_handle: Some(join_handle),
        })
    }

    /// Queue without blocking inference. Replaces an older not-yet-started
    /// checkpoint so graceful shutdown persists the newest available state.
    fn enqueue(&self, job: Qwen35DiskWriteJob) -> bool {
        let (lock, ready) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.shutdown {
            return false;
        }
        let replaced = state.pending.replace(job).is_some();
        debug_assert_eq!(QWEN35_ASYNC_PENDING_SLOTS, 1);
        ready.notify_one();
        replaced
    }

    fn pending_jobs(&self) -> usize {
        let (lock, _) = &*self.shared;
        let state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        usize::from(state.in_flight) + usize::from(state.pending.is_some())
    }

    #[cfg(test)]
    fn wait_for_idle(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let (lock, idle) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while state.in_flight || state.pending.is_some() {
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            let (next, wait_result) = idle
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if wait_result.timed_out() && (state.in_flight || state.pending.is_some()) {
                return false;
            }
        }
        true
    }
}

impl Drop for Qwen35AsyncWriter {
    fn drop(&mut self) {
        let pending_at_shutdown = self.pending_jobs();
        let started = Instant::now();
        let (lock, ready) = &*self.shared;
        {
            let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            state.shutdown = true;
            ready.notify_all();
        }
        let joined = self
            .join_handle
            .take()
            .map(|handle| handle.join().is_ok())
            .unwrap_or(true);
        tracing::info!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            pending_at_shutdown,
            drain_ms = started.elapsed().as_secs_f64() * 1_000.0,
            joined,
            "Qwen35 async checkpoint writer drained"
        );
    }
}

fn qwen35_async_writer_loop(
    target: &Qwen35DiskPersistor,
    shared: &Arc<(Mutex<Qwen35AsyncWriterState>, Condvar)>,
) {
    loop {
        let job = {
            let (lock, ready) = &**shared;
            let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            loop {
                if state.pending.is_none() {
                    if state.shutdown {
                        return;
                    }
                    state = ready
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    continue;
                }
                if state.shutdown {
                    break;
                }
                if state.active_requests > 0 {
                    state = ready
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    continue;
                }
                let idle_for = state
                    .idle_since
                    .map(|since| since.elapsed())
                    .unwrap_or_default();
                if idle_for >= QWEN35_ASYNC_IDLE_GRACE {
                    break;
                }
                let remaining = QWEN35_ASYNC_IDLE_GRACE - idle_for;
                let (next, _) = ready
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state = next;
            }
            match state.pending.take() {
                Some(job) => {
                    state.in_flight = true;
                    job
                }
                None => break,
            }
        };

        if let Err(error) = target.write(
            &job.cfg,
            &job.lcp_key_hex,
            job.snapshot.as_ref(),
            &job.sidecar,
        ) {
            tracing::warn!(
                target: "hf2q::serve::api::engine_qwen35::progress",
                cache_dir = %target.cache_dir().display(),
                key_hex = %job.lcp_key_hex,
                error = %format!("{error:#}"),
                "Qwen35 async checkpoint write failed; continuing"
            );
        }

        let (lock, idle) = &**shared;
        let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.in_flight = false;
        idle.notify_all();
    }
}

/// Keeps disk serialization outside an active Qwen request. The guard owns an
/// Arc, so it does not borrow `Qwen35LoadedModel` across generation.
pub struct Qwen35DiskRequestGuard {
    persistor: Arc<Qwen35DiskPersistor>,
}

impl Drop for Qwen35DiskRequestGuard {
    fn drop(&mut self) {
        self.persistor.finish_request();
    }
}

/// Disk-backed cold-resume persistor for qwen35 hybrid snapshots.
///
/// Constructed once per `cache_dir` at engine load; the `cache_dir`
/// argument is sourced from `HF2Q_KV_PERSIST` env (iter-6).
/// `Qwen35HybridConfig` is passed PER-CALL on `write` / `read` /
/// `hydrate_for_cfg` because cache shape (`max_seq_len`, `n_seqs`)
/// is per-prefill, not load-time (kv_cache.rs:347+ — see ADR-027
/// §4.7 iter-6a finding).
///
/// Multiple `Qwen35LoadedModel` instances using the same `cache_dir`
/// can safely share one persistor; per-cfg isolation is enforced by
/// the fingerprint subdir layout
/// (`<cache_dir>/<cfg-fingerprint>/<lcp_key>.bin`) — incompatible model
/// dimensions or substrate drift re-route to a different subdir, while
/// ordinary per-request `max_seq_len` changes share one compatible v5
/// namespace.
pub struct Qwen35DiskPersistor {
    /// Root cache directory (operator-controlled). Persisted files land
    /// under `<cache_dir>/<cfg-fingerprint>/<lcp_key>.bin`.
    cache_dir: PathBuf,
    /// ADR-027 sub-iter 23d-γ — on-disk byte budget per cfg-fingerprint
    /// subdir, enforced LRU-by-mtime after every successful `write`.
    /// `0` = unlimited (back-compat default for `new`). Production
    /// serve wires `HF2Q_KV_PERSIST_BUDGET_BYTES` through
    /// [`Self::new_with_budget`].
    ///
    /// Rationale: before this knob, a single ~100K-token agentic
    /// session wrote ~105 GB of chunk snapshots unbudgeted (measured
    /// live 2026-08-03), which is an operational disk-exhaustion hazard.
    budget_bytes: u64,
    /// Latest-wins, single-pending-slot writer. `None` only inside the
    /// background thread's synchronous target to avoid recursive workers.
    async_writer: Option<Qwen35AsyncWriter>,
}

impl Qwen35DiskPersistor {
    /// Construct a persistor rooted at `cache_dir`. Does NOT create
    /// the directory yet — the per-cfg subdir is created lazily on
    /// the first `write` for that cfg.
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        Self::new_with_budget(cache_dir, 0)
    }

    /// [`Self::new`] + an explicit on-disk byte budget (0 = unlimited).
    pub fn new_with_budget(cache_dir: PathBuf, budget_bytes: u64) -> Result<Self> {
        // Validate the parent dir is creatable; do not pre-create
        // per-cfg subdirs (we don't know which cfgs will be used yet).
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Qwen35DiskPersistor: mkdir {}", cache_dir.display()))?;
        let async_writer = Qwen35AsyncWriter::spawn(cache_dir.clone(), budget_bytes)?;
        Ok(Self {
            cache_dir,
            budget_bytes,
            async_writer: Some(async_writer),
        })
    }

    /// ADR-027 sub-iter 23d-γ — enforce `budget_bytes` within one
    /// cfg-fingerprint subdir: evict oldest-mtime `.bin` files until the
    /// subdir totals ≤ budget. The just-written file is never evicted
    /// (even when it alone exceeds the budget — a too-big latest
    /// snapshot is kept so the most recent resume point always
    /// survives; the next write re-evaluates). Eviction failures are
    /// warn-logged and swallowed (persistence-layer failure must not
    /// break inference).
    fn enforce_budget(&self, cfg_dir: &Path, keep: &Path) {
        if self.budget_bytes == 0 {
            return;
        }
        let read = match fs::read_dir(cfg_dir) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    dir = %cfg_dir.display(),
                    error = %e,
                    "ADR-027 23d-γ: budget scan failed; skipping eviction"
                );
                return;
            }
        };
        let mut files: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("bin") {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            files.push((path, mtime, meta.len()));
        }
        let mut total: u64 = files.iter().map(|f| f.2).sum();
        if total <= self.budget_bytes {
            return;
        }
        // Oldest mtime first; keep the just-written file regardless.
        files.sort_by_key(|f| f.1);
        for (path, _, size) in files {
            if total <= self.budget_bytes {
                break;
            }
            if path == keep {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => {
                    total = total.saturating_sub(size);
                    tracing::info!(
                        evicted = %path.display(),
                        freed_bytes = size,
                        remaining_bytes = total,
                        budget_bytes = self.budget_bytes,
                        "ADR-027 23d-γ: LRU evicted qwen35 LCP snapshot (budget)"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "ADR-027 23d-γ: budget eviction failed; continuing"
                    );
                }
            }
        }
    }

    /// Path to the file that would back `(cfg, lcp_key)`, regardless
    /// of whether the file currently exists on disk.
    pub fn path_for_key(&self, cfg: &Qwen35HybridConfig, lcp_key_hex: &str) -> PathBuf {
        let fingerprint_hex = compute_config_fingerprint_hex(cfg);
        self.cache_dir
            .join(fingerprint_hex)
            .join(format!("{lcp_key_hex}.bin"))
    }

    /// Serialize `snapshot` + `sidecar` through the QH35 envelope (under
    /// `cfg`) and write the bytes to
    /// `<cache_dir>/<cfg-fingerprint>/<lcp_key>.bin` atomically
    /// (tempfile + rename). Creates the per-cfg subdir on first use.
    /// Overwrites any existing file at the same key.
    ///
    /// `sidecar` carries the in-memory `LcpRegistry::store(...)` arguments
    /// (key fields, prompt_tokens, sliding_window, linear_capacity) that
    /// the cold-start hydrate path replays via
    /// `Qwen35LoadedModel::hydrate_lcp_registry_from_disk`. The caller
    /// constructs it from the live request state.
    pub fn write(
        &self,
        cfg: &Qwen35HybridConfig,
        lcp_key_hex: &str,
        snapshot: &HybridKvCacheSnapshot,
        sidecar: &LcpSidecarMetadata,
    ) -> Result<()> {
        ensure!(
            !lcp_key_hex.is_empty(),
            "Qwen35DiskPersistor::write: empty lcp_key_hex"
        );
        ensure!(
            lcp_key_hex
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "Qwen35DiskPersistor::write: lcp_key_hex must be alphanumeric/_/- only \
             (got {lcp_key_hex:?})"
        );
        let bytes = serialize_hybrid_with_sidecar(snapshot, cfg, sidecar)
            .context("Qwen35DiskPersistor::write: serialize_hybrid_with_sidecar")?;
        let final_path = self.path_for_key(cfg, lcp_key_hex);
        let cfg_dir = final_path
            .parent()
            .ok_or_else(|| anyhow!("Qwen35DiskPersistor::write: path has no parent"))?;
        fs::create_dir_all(cfg_dir).with_context(|| {
            format!(
                "Qwen35DiskPersistor::write: per-cfg mkdir {}",
                cfg_dir.display()
            )
        })?;
        let tmp_path = final_path.with_extension("bin.tmp");
        {
            let mut tmp = fs::File::create(&tmp_path).with_context(|| {
                format!("Qwen35DiskPersistor::write: create {}", tmp_path.display())
            })?;
            tmp.write_all(&bytes).with_context(|| {
                format!(
                    "Qwen35DiskPersistor::write: write_all {}",
                    tmp_path.display()
                )
            })?;
            tmp.sync_all().with_context(|| {
                format!("Qwen35DiskPersistor::write: sync {}", tmp_path.display())
            })?;
        }
        fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "Qwen35DiskPersistor::write: rename {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        // ADR-027 sub-iter 23d-γ: enforce the on-disk budget after every
        // successful write (no-op when budget = 0).
        self.enforce_budget(cfg_dir, &final_path);
        Ok(())
    }

    /// Submit a checkpoint without putting serialization, file I/O, or
    /// `fsync` on the inference thread. At most one older pending checkpoint
    /// is retained while the worker writes; a newer submission replaces it.
    /// The in-flight write is never interrupted, and dropping the persistor
    /// drains the newest pending job before joining the worker.
    ///
    /// Returns `true` when an older pending job was replaced. Returns an error
    /// only when shutdown has begun; persistence failure inside the worker is
    /// logged and does not affect the live in-memory cache.
    pub fn enqueue_write(
        &self,
        cfg: Qwen35HybridConfig,
        lcp_key_hex: String,
        snapshot: Arc<HybridKvCacheSnapshot>,
        sidecar: LcpSidecarMetadata,
    ) -> Result<bool> {
        let writer = self.async_writer.as_ref().ok_or_else(|| {
            anyhow!("Qwen35DiskPersistor::enqueue_write: async writer is unavailable")
        })?;
        let shutting_down = {
            let (lock, _) = &*writer.shared;
            let state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            state.shutdown
        };
        ensure!(
            !shutting_down,
            "Qwen35DiskPersistor::enqueue_write: writer is shutting down"
        );
        Ok(writer.enqueue(Qwen35DiskWriteJob {
            cfg,
            lcp_key_hex,
            snapshot,
            sidecar,
        }))
    }

    /// Number of accepted writes currently running or pending. Intended for
    /// verbose operator feedback and shutdown diagnostics.
    pub fn pending_writes(&self) -> usize {
        self.async_writer
            .as_ref()
            .map(Qwen35AsyncWriter::pending_jobs)
            .unwrap_or(0)
    }

    /// Mark the beginning of a Qwen request. The returned guard clears the
    /// activity mark on every exit path, including cancellation and errors.
    pub fn begin_request(self: &Arc<Self>) -> Qwen35DiskRequestGuard {
        if let Some(writer) = self.async_writer.as_ref() {
            let (lock, _) = &*writer.shared;
            let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            state.active_requests = state.active_requests.saturating_add(1);
            state.idle_since = None;
        }
        Qwen35DiskRequestGuard {
            persistor: Arc::clone(self),
        }
    }

    fn finish_request(&self) {
        if let Some(writer) = self.async_writer.as_ref() {
            let (lock, ready) = &*writer.shared;
            let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            state.active_requests = state.active_requests.saturating_sub(1);
            if state.active_requests == 0 {
                state.idle_since = Some(Instant::now());
                ready.notify_all();
            }
        }
    }

    #[cfg(test)]
    fn wait_for_async_idle(&self, timeout: std::time::Duration) -> bool {
        self.async_writer
            .as_ref()
            .map(|writer| writer.wait_for_idle(timeout))
            .unwrap_or(true)
    }

    /// Read the snapshot + sidecar metadata keyed by `(cfg, lcp_key_hex)`
    /// from disk; returns `Ok(None)` when the file doesn't exist (clean
    /// cache miss). Returns `Err` on file-present-but-corrupt (caller
    /// should treat as a cache miss + delete + log; iter-6b's wire-up
    /// implements that recovery).
    ///
    /// The returned tuple is `(snapshot, sidecar)`; the sidecar carries
    /// the original `LcpRegistry::store(...)` arguments needed for
    /// hydrate-time re-insertion.
    pub fn read(
        &self,
        cfg: &Qwen35HybridConfig,
        lcp_key_hex: &str,
        device: &MlxDevice,
    ) -> Result<Option<(HybridKvCacheSnapshot, LcpSidecarMetadata)>> {
        let path = self.path_for_key(cfg, lcp_key_hex);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(anyhow!(
                    "Qwen35DiskPersistor::read: open {}: {e}",
                    path.display()
                ))
            }
        };
        let pair = deserialize_hybrid_with_sidecar(&bytes, cfg, device).with_context(|| {
            format!(
                "Qwen35DiskPersistor::read: deserialize {} ({} bytes)",
                path.display(),
                bytes.len()
            )
        })?;
        Ok(Some(pair))
    }

    /// Walk `<cache_dir>/<cfg-fingerprint>/` and return every
    /// `(lcp_key, snapshot, sidecar)` triple that successfully
    /// deserializes against `cfg`. Files that fail to deserialize
    /// (corrupt / wrong-config / drift) are skipped with a warning
    /// trace — they don't poison the hydrate.
    ///
    /// Iter-6b.3 calls this at first prefill (when cfg is known) to
    /// seed the in-memory `LcpRegistry` for that cfg via the
    /// sidecar's `(LcpKey, prompt_tokens, sliding_window,
    /// linear_capacity)` fields. Earlier-cfg snapshots (under
    /// different fingerprint subdirs) are NOT hydrated — they live
    /// correctly under their own subdir until a matching-cfg prefill
    /// arrives.
    pub fn hydrate_for_cfg(
        &self,
        cfg: &Qwen35HybridConfig,
        device: &MlxDevice,
    ) -> Result<Vec<(String, HybridKvCacheSnapshot, LcpSidecarMetadata)>> {
        let fingerprint_hex = compute_config_fingerprint_hex(cfg);
        let dir = self.cache_dir.join(&fingerprint_hex);
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Empty cfg-subdir is a clean cold start — return [].
                return Ok(Vec::new());
            }
            Err(e) => {
                return Err(anyhow!(
                    "Qwen35DiskPersistor::hydrate_for_cfg: read_dir {}: {e}",
                    dir.display()
                ))
            }
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| {
                format!(
                    "Qwen35DiskPersistor::hydrate_for_cfg: dir entry under {}",
                    dir.display()
                )
            })?;
            let path = entry.path();
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if !name.ends_with(".bin") || name.ends_with(".tmp") {
                continue;
            }
            let lcp_key_hex = name.trim_end_matches(".bin").to_string();
            match self.read(cfg, &lcp_key_hex, device) {
                Ok(Some((snap, sidecar))) => out.push((lcp_key_hex, snap, sidecar)),
                Ok(None) => {
                    // Race window: file vanished between read_dir and
                    // open. Skip silently.
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Qwen35DiskPersistor::hydrate_for_cfg: skipping corrupt cache file"
                    );
                }
            }
        }
        Ok(out)
    }

    /// `cache_dir` getter — used by iter-6b wire-up + tests.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Compute the cfg-fingerprint subdir name (16-hex chars) for an
    /// arbitrary `cfg` — exposed for tests + log lines that want to
    /// surface "which cfg-subdir is in use" without exposing the full
    /// hashing details.
    pub fn fingerprint_hex_for(cfg: &Qwen35HybridConfig) -> String {
        compute_config_fingerprint_hex(cfg)
    }
}

/// Compute the 8-byte (16-hex-char) compatibility fingerprint of a
/// `Qwen35HybridConfig`.
///
/// QH35 v5 snapshots may compact axis 2 (sequence capacity) to the stored
/// prefix. The fingerprint therefore excludes axis 2 from full-attention
/// and MTP shapes while retaining every non-sequence dimension, layer count,
/// sequence count, codec, fixed DeltaNet state shape, and substrate. This is
/// what lets a checkpoint written by one OpenCode turn hydrate after restart
/// into the differently-sized cache allocated for the next turn.
fn compute_config_fingerprint_hex(cfg: &Qwen35HybridConfig) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"QH35-cfg-fp-v2-capacity-independent");
    h.update(&cfg.n_full_attn.to_le_bytes());
    h.update(&cfg.n_linear_attn.to_le_bytes());
    h.update(&[u8::from(cfg.has_mtp)]);
    h.update(&cfg.n_seqs.to_le_bytes());
    for &dim in &[
        cfg.full_attn_shape[0],
        cfg.full_attn_shape[1],
        cfg.full_attn_shape[3],
    ] {
        h.update(&dim.to_le_bytes());
    }
    h.update(&[full_attn_codec_byte(cfg.full_attn_codec)]);
    for &dim in &cfg.linear_conv_shape {
        h.update(&dim.to_le_bytes());
    }
    for &dim in &cfg.linear_recurrent_shape {
        h.update(&dim.to_le_bytes());
    }
    for &dim in &[cfg.mtp_shape[0], cfg.mtp_shape[1], cfg.mtp_shape[3]] {
        h.update(&dim.to_le_bytes());
    }
    // ADR-027 sub-iter 23d-γ: mix the KV substrate into the fingerprint
    // so a snapshot written under one substrate (F32-only / TQ-only /
    // Both) never hydrates into a cache allocated under another — the
    // mismatched (snapshot, cache) substrate pairs would restore nothing
    // and leave the resumed prefix zeroed (silent coherence corruption).
    // Same-process mode is fixed at startup, so same-mode writes and
    // reads still collide exactly as before.
    h.update(&[cfg.kv_substrate as u8]);
    let digest = h.finalize();
    hex::encode(&digest[..8])
}

fn full_attn_codec_byte(codec: FullAttnCodec) -> u8 {
    match codec {
        FullAttnCodec::F32Dense => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::kv_cache::MtpKvSnapshot;
    use crate::serve::kv_persist::families::qwen35_hybrid_persistor::KvSubstrate;
    use crate::serve::kv_persist::format::ModelFingerprint;
    use mlx_native::DType;

    /// Synth sidecar metadata for disk-persistor unit tests. The fields
    /// are deterministic so failure modes (drift, truncation) surface
    /// against expected bytes; production builds a sidecar from the
    /// live `LcpKey` + cache state in `store_lcp_with_disk_writeback`.
    fn synth_sidecar() -> LcpSidecarMetadata {
        let mut fp = [0u8; 32];
        for (i, b) in fp.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        LcpSidecarMetadata {
            model_fingerprint: ModelFingerprint(fp),
            tenant_id: "default".to_string(),
            params_hash: 0xCAFE_BABE_DEAD_BEEF,
            prompt_tokens: vec![10, 20, 30, 40, 50, 60, 70, 80],
            sliding_window: 1024,
            linear_capacity: 512,
        }
    }

    /// Standalone synth-snapshot helper for the disk-persistor tests
    /// (avoids cross-module test imports). Mirrors the pattern in
    /// qwen35_hybrid_persistor::tests::synth_full_attn_only_snapshot.
    fn synth_snapshot(device: &MlxDevice, cfg: &Qwen35HybridConfig) -> HybridKvCacheSnapshot {
        let elems_full: usize = cfg.full_attn_shape.iter().product::<u64>() as usize;
        let bytes_full = elems_full * std::mem::size_of::<f32>();
        let shape_full: Vec<usize> = cfg.full_attn_shape.iter().map(|d| *d as usize).collect();

        let mut full_attn_k = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_v = Vec::with_capacity(cfg.n_full_attn as usize);
        let mut full_attn_current_len = Vec::with_capacity(cfg.n_full_attn as usize);
        for slot in 0..cfg.n_full_attn as usize {
            let mut k = device
                .alloc_buffer(bytes_full, DType::F32, shape_full.clone())
                .unwrap();
            {
                let dst = k.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 31 + i) % 251) as u8;
                }
            }
            let mut v = device
                .alloc_buffer(bytes_full, DType::F32, shape_full.clone())
                .unwrap();
            {
                let dst = v.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 37 + i) % 251) as u8;
                }
            }
            // ADR-027 sub-sub-iter 23a-β: test fixture wraps in Some.
            full_attn_k.push(Some(k));
            full_attn_v.push(Some(v));
            full_attn_current_len.push((0..cfg.n_seqs).map(|s| (slot as u32) + s).collect());
        }

        let elems_conv: usize = cfg.linear_conv_shape.iter().product::<u64>() as usize;
        let bytes_conv = elems_conv * std::mem::size_of::<f32>();
        let shape_conv: Vec<usize> = cfg.linear_conv_shape.iter().map(|d| *d as usize).collect();
        let elems_rec: usize = cfg.linear_recurrent_shape.iter().product::<u64>() as usize;
        let bytes_rec = elems_rec * std::mem::size_of::<f32>();
        let shape_rec: Vec<usize> = cfg
            .linear_recurrent_shape
            .iter()
            .map(|d| *d as usize)
            .collect();

        let mut linear_conv = Vec::new();
        let mut linear_recurrent = Vec::new();
        for slot in 0..cfg.n_linear_attn as usize {
            let mut c = device
                .alloc_buffer(bytes_conv, DType::F32, shape_conv.clone())
                .unwrap();
            {
                let dst = c.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 41 + i) % 251) as u8;
                }
            }
            let mut r = device
                .alloc_buffer(bytes_rec, DType::F32, shape_rec.clone())
                .unwrap();
            {
                let dst = r.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((slot * 43 + i) % 251) as u8;
                }
            }
            linear_conv.push(c);
            linear_recurrent.push(r);
        }

        let mtp = if cfg.has_mtp {
            let elems: usize = cfg.mtp_shape.iter().product::<u64>() as usize;
            let bytes_len = elems * std::mem::size_of::<f32>();
            let shape: Vec<usize> = cfg.mtp_shape.iter().map(|d| *d as usize).collect();
            let mut k = device
                .alloc_buffer(bytes_len, DType::F32, shape.clone())
                .unwrap();
            {
                let dst = k.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((47 * i + 11) % 251) as u8;
                }
            }
            let mut v = device.alloc_buffer(bytes_len, DType::F32, shape).unwrap();
            {
                let dst = v.as_mut_slice::<u8>().unwrap();
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((53 * i + 13) % 251) as u8;
                }
            }
            Some(MtpKvSnapshot {
                // ADR-027 sub-sub-iter 23a-α: test fixture wraps in Some.
                k: Some(k),
                v: Some(v),
                current_len: (0..cfg.n_seqs).map(|s| 100 + s).collect(),
                // iter-35 (sub-iter 23d-α): test fixture, no TQ.
                tq: None,
            })
        } else {
            None
        };

        let n_full_attn = full_attn_k.len();
        HybridKvCacheSnapshot {
            full_attn_k,
            full_attn_v,
            full_attn_current_len,
            // iter-35 (sub-iter 23d-α): test fixture, no TQ.
            full_attn_tq: (0..n_full_attn).map(|_| None).collect(),
            mtp,
            linear_conv,
            linear_recurrent,
        }
    }

    fn synth_cfg() -> Qwen35HybridConfig {
        Qwen35HybridConfig {
            n_full_attn: 2,
            n_linear_attn: 3,
            has_mtp: true,
            n_seqs: 1,
            full_attn_shape: [1, 2, 8, 4],
            full_attn_codec: FullAttnCodec::F32Dense,
            linear_conv_shape: [4, 3, 1],
            linear_recurrent_shape: [4, 8, 2, 1],
            mtp_shape: [1, 4, 8, 4],
            kv_substrate: KvSubstrate::F32Only,
        }
    }

    fn snapshots_byte_equal(a: &HybridKvCacheSnapshot, b: &HybridKvCacheSnapshot) -> bool {
        if a.full_attn_k.len() != b.full_attn_k.len() {
            return false;
        }
        if a.full_attn_v.len() != b.full_attn_v.len() {
            return false;
        }
        if a.full_attn_current_len != b.full_attn_current_len {
            return false;
        }
        for i in 0..a.full_attn_k.len() {
            // ADR-027 sub-sub-iter 23a-β: Optional full-attn K/V — compare
            // Some-to-Some byte-equal.
            if a.full_attn_k[i]
                .as_ref()
                .expect("a.k some")
                .as_slice::<u8>()
                .unwrap()
                != b.full_attn_k[i]
                    .as_ref()
                    .expect("b.k some")
                    .as_slice::<u8>()
                    .unwrap()
            {
                return false;
            }
            if a.full_attn_v[i]
                .as_ref()
                .expect("a.v some")
                .as_slice::<u8>()
                .unwrap()
                != b.full_attn_v[i]
                    .as_ref()
                    .expect("b.v some")
                    .as_slice::<u8>()
                    .unwrap()
            {
                return false;
            }
        }
        if a.linear_conv.len() != b.linear_conv.len() {
            return false;
        }
        for i in 0..a.linear_conv.len() {
            if a.linear_conv[i].as_slice::<u8>().unwrap()
                != b.linear_conv[i].as_slice::<u8>().unwrap()
            {
                return false;
            }
            if a.linear_recurrent[i].as_slice::<u8>().unwrap()
                != b.linear_recurrent[i].as_slice::<u8>().unwrap()
            {
                return false;
            }
        }
        match (&a.mtp, &b.mtp) {
            (None, None) => true,
            (Some(am), Some(bm)) => {
                // ADR-027 sub-sub-iter 23a-α: Optional MTP K/V — compare
                // Some-to-Some byte-equal (None-to-None test fixture path
                // not exercised today; iter-23c+ adds it).
                am.k.as_ref().expect("am.k some").as_slice::<u8>().unwrap()
                    == bm.k.as_ref().expect("bm.k some").as_slice::<u8>().unwrap()
                    && am.v.as_ref().expect("am.v some").as_slice::<u8>().unwrap()
                        == bm.v.as_ref().expect("bm.v some").as_slice::<u8>().unwrap()
                    && am.current_len == bm.current_len
            }
            _ => false,
        }
    }

    /// Make a unique tempdir per test so parallel test execution
    /// doesn't collide.
    fn unique_tempdir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "qh35-disk-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn qh35_disk_round_trip_via_tempdir() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("rt");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        persistor
            .write(&cfg, "deadbeef_cafebabe", &snap, &sidecar)
            .unwrap();
        let (restored_snap, restored_sidecar) = persistor
            .read(&cfg, "deadbeef_cafebabe", &device)
            .unwrap()
            .expect("write+read should round-trip");
        assert!(snapshots_byte_equal(&snap, &restored_snap));
        assert_eq!(restored_sidecar, sidecar);
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_async_write_lands_and_round_trips() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("async-rt");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snapshot = Arc::new(synth_snapshot(&device, &cfg));
        let sidecar = synth_sidecar();
        persistor
            .enqueue_write(
                cfg.clone(),
                "async_key".to_string(),
                Arc::clone(&snapshot),
                sidecar.clone(),
            )
            .expect("async enqueue");
        assert!(
            persistor.wait_for_async_idle(std::time::Duration::from_secs(5)),
            "async writer did not drain within five seconds"
        );
        let (restored, restored_sidecar) = persistor
            .read(&cfg, "async_key", &device)
            .expect("async read")
            .expect("async write must land");
        assert!(snapshots_byte_equal(snapshot.as_ref(), &restored));
        assert_eq!(restored_sidecar, sidecar);
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_async_drop_drains_newest_pending_checkpoint() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("async-drop");
        let _ = fs::remove_dir_all(&tempdir);
        let snapshot = Arc::new(synth_snapshot(&device, &cfg));
        let sidecar = synth_sidecar();
        {
            let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
            for key in ["async_old", "async_middle", "async_newest"] {
                persistor
                    .enqueue_write(
                        cfg.clone(),
                        key.to_string(),
                        Arc::clone(&snapshot),
                        sidecar.clone(),
                    )
                    .expect("async enqueue before drop");
            }
            // Drop closes submission, drains the newest pending checkpoint,
            // and joins the worker before this scope exits.
        }

        let reader = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let restored = reader
            .read(&cfg, "async_newest", &device)
            .expect("newest read after drain")
            .expect("newest pending checkpoint must survive graceful drop");
        assert!(snapshots_byte_equal(snapshot.as_ref(), &restored.0));
        assert_eq!(restored.1, sidecar);
        drop(reader);
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_async_writer_waits_until_request_is_idle() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("async-idle");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Arc::new(Qwen35DiskPersistor::new(tempdir.clone()).unwrap());
        let request = persistor.begin_request();
        persistor
            .enqueue_write(
                cfg.clone(),
                "idle_key".to_string(),
                Arc::new(synth_snapshot(&device, &cfg)),
                synth_sidecar(),
            )
            .expect("enqueue during active request");
        std::thread::sleep(QWEN35_ASYNC_IDLE_GRACE + Duration::from_millis(50));
        assert!(
            !persistor.path_for_key(&cfg, "idle_key").exists(),
            "disk serialization must not overlap an active request"
        );
        assert_eq!(persistor.pending_writes(), 1);

        drop(request);
        assert!(
            persistor.wait_for_async_idle(Duration::from_secs(5)),
            "writer did not drain after request became idle"
        );
        assert!(persistor.path_for_key(&cfg, "idle_key").exists());
        drop(persistor);
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_read_missing_returns_none() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("miss");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let result = persistor.read(&cfg, "nonexistent_key", &device).unwrap();
        assert!(result.is_none());
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_hydrate_for_cfg_collects_every_file() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("hydrate");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        let keys = ["key_a", "key_b", "key_c"];
        for k in &keys {
            persistor.write(&cfg, k, &snap, &sidecar).unwrap();
        }
        let mut hydrated = persistor.hydrate_for_cfg(&cfg, &device).unwrap();
        hydrated.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(hydrated.len(), keys.len());
        for (got, expected) in hydrated.iter().zip(keys.iter()) {
            assert_eq!(&got.0, expected);
            assert!(snapshots_byte_equal(&snap, &got.1));
            assert_eq!(got.2, sidecar);
        }
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_fingerprint_subdir_isolates_cfg_drift() {
        // Two configs with different shapes routed to different
        // fingerprint subdirs by ONE shared persistor — a key written
        // under cfg-A must NOT appear under cfg-B.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg_a = synth_cfg();
        let mut cfg_b = synth_cfg();
        cfg_b.full_attn_shape = [1, 4, 8, 4]; // n_kv_heads=4 instead of 2
        let tempdir = unique_tempdir("fp");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        assert_ne!(
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_a),
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_b),
            "differently-shaped configs MUST produce different fingerprints"
        );
        let snap_a = synth_snapshot(&device, &cfg_a);
        let sidecar = synth_sidecar();
        persistor
            .write(&cfg_a, "shared_key", &snap_a, &sidecar)
            .unwrap();
        // Reading the SAME key under cfg_b targets a different subdir
        // → must miss.
        let result_b = persistor.read(&cfg_b, "shared_key", &device).unwrap();
        assert!(
            result_b.is_none(),
            "cfg_b read must not see cfg_a's snapshot under shared key"
        );
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_v5_fingerprint_ignores_only_request_sequence_capacity() {
        let cfg_a = synth_cfg();
        let mut cfg_b = synth_cfg();
        cfg_b.full_attn_shape[2] = cfg_a.full_attn_shape[2] + 4096;
        cfg_b.mtp_shape[2] = cfg_a.mtp_shape[2] + 4096;
        assert_eq!(
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_a),
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_b),
            "ordinary request-capacity growth must share the compact-v5 cache namespace"
        );

        cfg_b.full_attn_shape[3] += 1;
        assert_ne!(
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_a),
            Qwen35DiskPersistor::fingerprint_hex_for(&cfg_b),
            "head-dimension drift remains incompatible"
        );
    }

    #[test]
    fn qh35_v5_disk_replays_compact_snapshot_into_larger_request_capacity() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg_small = synth_cfg();
        let mut cfg_large = synth_cfg();
        cfg_large.full_attn_shape[2] = cfg_small.full_attn_shape[2] + 8;
        cfg_large.mtp_shape[2] = cfg_small.mtp_shape[2] + 8;

        let tempdir = unique_tempdir("cross-capacity-v5");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snapshot = synth_snapshot(&device, &cfg_small);
        let sidecar = synth_sidecar();
        persistor
            .write(&cfg_small, "stable_turn", &snapshot, &sidecar)
            .unwrap();

        let (restored, restored_sidecar) = persistor
            .read(&cfg_large, "stable_turn", &device)
            .expect("larger-capacity read")
            .expect("capacity-independent fingerprint must address the v5 file");
        assert!(snapshots_byte_equal(&snapshot, &restored));
        assert_eq!(restored_sidecar, sidecar);
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_atomic_overwrite_replaces_prior_content() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("overwrite");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        // Write snap-A, then overwrite with mutated snap-B.
        let snap_a = synth_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        persistor.write(&cfg, "the_key", &snap_a, &sidecar).unwrap();
        let mut snap_b = synth_snapshot(&device, &cfg);
        {
            // ADR-027 sub-sub-iter 23a-β: Optional full-attn K/V.
            let dst = snap_b.full_attn_k[0]
                .as_mut()
                .expect("snap_b.k[0] some")
                .as_mut_slice::<u8>()
                .unwrap();
            for b in dst.iter_mut() {
                *b = b.wrapping_add(7);
            }
        }
        persistor.write(&cfg, "the_key", &snap_b, &sidecar).unwrap();
        let (restored_snap, _) = persistor.read(&cfg, "the_key", &device).unwrap().unwrap();
        assert!(snapshots_byte_equal(&snap_b, &restored_snap));
        assert!(!snapshots_byte_equal(&snap_a, &restored_snap));
        let _ = fs::remove_dir_all(&tempdir);
    }

    #[test]
    fn qh35_disk_multi_cfg_cohabit_one_persistor() {
        // Iter-6b API affordance: a single persistor handles multiple
        // cfgs (e.g., two prefills with different max_seq_len). Each
        // cfg lives in its own fingerprint subdir; reads + writes
        // route correctly per cfg.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg_a = synth_cfg();
        // cfg_b: distinct full_attn_shape (different max_seq_len would
        // be the production drift; we use a different head count so
        // synth_snapshot produces a structurally-different cache).
        let mut cfg_b = synth_cfg();
        cfg_b.full_attn_shape = [1, 4, 8, 4];
        let tempdir = unique_tempdir("multi-cfg");
        let _ = fs::remove_dir_all(&tempdir);
        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap_a = synth_snapshot(&device, &cfg_a);
        let snap_b = synth_snapshot(&device, &cfg_b);
        let sidecar = synth_sidecar();
        persistor.write(&cfg_a, "key1", &snap_a, &sidecar).unwrap();
        persistor.write(&cfg_b, "key1", &snap_b, &sidecar).unwrap();
        // Each cfg sees ITS OWN snap at the same key.
        let (r_a, _) = persistor.read(&cfg_a, "key1", &device).unwrap().unwrap();
        let (r_b, _) = persistor.read(&cfg_b, "key1", &device).unwrap().unwrap();
        assert!(snapshots_byte_equal(&snap_a, &r_a));
        assert!(snapshots_byte_equal(&snap_b, &r_b));
        // Hydrate-for-cfg returns ONLY that cfg's entries.
        let h_a = persistor.hydrate_for_cfg(&cfg_a, &device).unwrap();
        let h_b = persistor.hydrate_for_cfg(&cfg_b, &device).unwrap();
        assert_eq!(h_a.len(), 1);
        assert_eq!(h_b.len(), 1);
        let _ = fs::remove_dir_all(&tempdir);
    }

    /// ADR-027 Phase A iter-6b.3 — cross-process replay. Proves the
    /// full cold-start hydrate semantic at the disk-persistor +
    /// LcpRegistry layer:
    ///
    /// 1. Persistor A writes a snapshot+sidecar at a real LcpKey.
    /// 2. Persistor A is dropped (simulates process crash/exit).
    /// 3. Persistor B opens the SAME cache_dir (fresh process state).
    /// 4. A fresh, empty LcpRegistry is constructed.
    /// 5. Persistor B's hydrate_for_cfg walks the dir + returns the
    ///    snapshot+sidecar.
    /// 6. The hydrate code-path replays `lcp_registry.store(key,
    ///    prompt_tokens, vec![Arc::new(snap)], sliding_window,
    ///    linear_capacity)` from the sidecar fields.
    /// 7. lcp_registry.lookup(&original_key, &new_tokens) returns Some
    ///    with the expected LCP length, proving the registry state was
    ///    reconstructed identically.
    ///
    /// This exercises the FULL contract: write codec + read codec +
    /// sidecar fidelity + LcpKey reconstruction + Arc-payload re-wrap
    /// + LcpRegistry::store + LcpRegistry::lookup. SERVE end-to-end
    /// (engine + GPU model + 2-turn HTTP) is gated on operator
    /// validation in iter-6b.3g.
    #[test]
    fn qh35_disk_cross_process_replay_into_fresh_lcp_registry() {
        use crate::serve::kv_persist::lcp_registry::{LcpKey, LcpRegistry};
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("xprocess");
        let _ = fs::remove_dir_all(&tempdir);

        // The original prompt — sidecar must round-trip every byte so
        // the new-process lookup can compute LCP exactly.
        let original_prompt: Vec<u32> =
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500];
        // The new request's prompt: shares an LCP of length 12 with the
        // original (then diverges). Lookup should report k=12.
        let new_request_prompt: Vec<u32> =
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 999, 999, 999];

        let mut fp = [0u8; 32];
        for (i, b) in fp.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(19).wrapping_add(7);
        }
        let original_key = LcpKey {
            model_fingerprint: ModelFingerprint(fp),
            tenant_id: "qwen35:lcp_chunk:64".to_string(),
            params_hash: 0xABCD_EF01_2345_6789,
        };
        let original_sliding_window: u64 = 8192;
        let original_linear_capacity: u64 = 4096;

        // ── Process A: write snapshot+sidecar to disk ──
        {
            let persistor_a = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
            let snap = synth_snapshot(&device, &cfg);
            let sidecar = LcpSidecarMetadata {
                model_fingerprint: original_key.model_fingerprint.clone(),
                tenant_id: original_key.tenant_id.clone(),
                params_hash: original_key.params_hash,
                prompt_tokens: original_prompt.clone(),
                sliding_window: original_sliding_window,
                linear_capacity: original_linear_capacity,
            };
            persistor_a
                .write(&cfg, "fixedkey0001", &snap, &sidecar)
                .unwrap();
        } // persistor_a dropped — simulates process exit

        // ── Process B: fresh persistor + fresh registry; hydrate ──
        let mut registry: LcpRegistry<HybridKvCacheSnapshot> = LcpRegistry::new(8);
        assert_eq!(registry.len(), 0, "fresh registry must be empty");

        let persistor_b = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let triples = persistor_b.hydrate_for_cfg(&cfg, &device).unwrap();
        assert_eq!(
            triples.len(),
            1,
            "hydrate_for_cfg should find the one snapshot persistor A wrote"
        );

        // Replay the same store-call shape that
        // `Qwen35LoadedModel::hydrate_lcp_registry_from_disk` performs.
        for (_key_hex, snap, sidecar) in triples {
            let key = LcpKey {
                model_fingerprint: sidecar.model_fingerprint.clone(),
                tenant_id: sidecar.tenant_id.clone(),
                params_hash: sidecar.params_hash,
            };
            registry
                .store(
                    key,
                    sidecar.prompt_tokens.clone(),
                    vec![std::sync::Arc::new(snap)],
                    sidecar.sliding_window as usize,
                    sidecar.linear_capacity as usize,
                )
                .expect("registry should accept the hydrated entry");
        }

        // ── The proof: lookup with the ORIGINAL key must hit, and the
        // returned prefix must report the expected LCP length + the
        // original sliding_window / linear_capacity ──
        let prefix = registry
            .lookup(&original_key, &new_request_prompt)
            .expect("post-hydrate lookup with original key must hit");
        assert_eq!(
            prefix.k, 12,
            "LCP length should match the shared prefix between original_prompt and new_request_prompt"
        );
        assert_eq!(prefix.sliding_window, original_sliding_window as usize);
        assert_eq!(prefix.linear_capacity, original_linear_capacity as usize);
        assert_eq!(prefix.cached_prompt_len, original_prompt.len());

        let _ = fs::remove_dir_all(&tempdir);
    }

    /// ADR-027 Phase B iter-44 — cross-process replay with TQ-populated
    /// snapshot.
    ///
    /// Sibling of `qh35_disk_cross_process_replay_into_fresh_lcp_registry`
    /// (which uses an F32-only synthetic snapshot). This test populates
    /// the snapshot's TQ buffers (mirroring iter-34's production case
    /// where slot.k=None, slot.tq=Some) and validates:
    /// (a) v3 codec serializes TQ bytes to disk (iter-36 path)
    /// (b) v3 codec deserializes TQ bytes back from disk in a fresh
    ///     process (iter-36 reader)
    /// (c) The hydrated `HybridKvCacheSnapshot.full_attn_tq` field
    ///     contains the same bytes the writer put in.
    ///
    /// Closes the last validation gap for the iter-23 chain: end-to-end
    /// disk persistence + cross-process replay in TQ-only mode is now
    /// proven by code+test (not just unit-level codec tests). Future
    /// production cold-start hydrate scenarios in TQ-only mode are
    /// regression-protected.
    #[test]
    fn qh35_disk_cross_process_replay_with_tq_payload_byte_equal() {
        use crate::inference::models::qwen35::kv_cache::TqKvSnapshot;
        use crate::serve::kv_persist::lcp_registry::{LcpKey, LcpRegistry};
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        // Use a minimal cfg (no linear-attn, no MTP) to keep the test
        // focused on the TQ codec round-trip path. The full
        // synth_cfg with linear-attn + MTP is exercised by the
        // sibling F32 cross-process test above.
        let cfg = Qwen35HybridConfig {
            n_full_attn: 2,
            n_linear_attn: 0,
            has_mtp: false,
            n_seqs: 1,
            full_attn_shape: [1, 2, 8, 4],
            full_attn_codec: FullAttnCodec::F32Dense,
            linear_conv_shape: [0, 0, 0],
            linear_recurrent_shape: [0, 0, 0, 0],
            mtp_shape: [0, 0, 0, 0],
            kv_substrate: KvSubstrate::TqOnly,
        };
        let tempdir = unique_tempdir("xprocess-tq");
        let _ = fs::remove_dir_all(&tempdir);

        // Build a TQ-populated snapshot (slot.k=None, slot.tq=Some).
        // We emit None for K/V (TQ-only mode) and Some for TQ.
        let n_full_attn = cfg.n_full_attn as usize;
        let mut full_attn_tq: Vec<Option<TqKvSnapshot>> = Vec::with_capacity(n_full_attn);
        for slot in 0..n_full_attn {
            // TQ packed: 64 bytes ([1,2,8,4] U8); norms: 16 f32 = 64 bytes.
            // cfg.full_attn_shape = [1,2,8,4] → packed 64 B; norms [1,2,8,1] F32 = 64 B
            // (sub-iter 23d-γ shape-validated deserialize requires cfg-derived lengths).
            let packed_shape = vec![1, 2, 8, 4];
            let norms_shape = vec![1, 2, 8, 1];
            let mut k_packed = device
                .alloc_buffer(64, DType::U8, packed_shape.clone())
                .unwrap();
            let mut k_norms = device
                .alloc_buffer(64, DType::F32, norms_shape.clone())
                .unwrap();
            let mut v_packed = device.alloc_buffer(64, DType::U8, packed_shape).unwrap();
            let mut v_norms = device.alloc_buffer(64, DType::F32, norms_shape).unwrap();
            for (i, b) in k_packed
                .as_mut_slice::<u8>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *b = ((slot * 41 + i * 13) % 251) as u8;
            }
            for (i, b) in v_packed
                .as_mut_slice::<u8>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *b = ((slot * 17 + i * 23) % 251) as u8;
            }
            for (i, f) in k_norms
                .as_mut_slice::<f32>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *f = (slot as f32) * 0.75 + (i as f32) * 0.25;
            }
            for (i, f) in v_norms
                .as_mut_slice::<f32>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *f = (slot as f32) * 0.5 + (i as f32) * 0.125;
            }
            full_attn_tq.push(Some(TqKvSnapshot {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
                norms_per_pos: 1,
            }));
        }
        let snap = HybridKvCacheSnapshot {
            full_attn_k: (0..n_full_attn).map(|_| None).collect(),
            full_attn_v: (0..n_full_attn).map(|_| None).collect(),
            full_attn_current_len: (0..n_full_attn)
                .map(|s| (0..cfg.n_seqs).map(|seq| s as u32 * 100 + seq).collect())
                .collect(),
            full_attn_tq,
            mtp: None,
            linear_conv: Vec::new(),
            linear_recurrent: Vec::new(),
        };

        let original_prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let new_request_prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 999, 999, 999];

        let mut fp = [0u8; 32];
        for (i, b) in fp.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(11);
        }
        let original_key = LcpKey {
            model_fingerprint: ModelFingerprint(fp),
            tenant_id: "qwen35:tq:test".to_string(),
            params_hash: 0xCAFE_F00D_BEEF_DEAD,
        };

        // ── Process A: write TQ-populated snapshot+sidecar to disk ──
        // Capture source TQ bytes BEFORE moving into write — needed for
        // post-hydrate byte-equality assertions.
        let mut src_k_packed: Vec<Vec<u8>> = Vec::with_capacity(n_full_attn);
        let mut src_v_packed: Vec<Vec<u8>> = Vec::with_capacity(n_full_attn);
        let mut src_k_norms: Vec<Vec<u8>> = Vec::with_capacity(n_full_attn);
        let mut src_v_norms: Vec<Vec<u8>> = Vec::with_capacity(n_full_attn);
        for tq in snap.full_attn_tq.iter() {
            let tq = tq.as_ref().unwrap();
            src_k_packed.push(tq.k_packed.as_slice::<u8>().unwrap().to_vec());
            src_v_packed.push(tq.v_packed.as_slice::<u8>().unwrap().to_vec());
            src_k_norms.push(tq.k_norms.as_slice::<u8>().unwrap().to_vec());
            src_v_norms.push(tq.v_norms.as_slice::<u8>().unwrap().to_vec());
        }
        {
            let persistor_a = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
            let sidecar = LcpSidecarMetadata {
                model_fingerprint: original_key.model_fingerprint.clone(),
                tenant_id: original_key.tenant_id.clone(),
                params_hash: original_key.params_hash,
                prompt_tokens: original_prompt.clone(),
                sliding_window: 4096,
                linear_capacity: 2048,
            };
            persistor_a
                .write(&cfg, "tqkey0001", &snap, &sidecar)
                .unwrap();
        } // persistor_a + snap dropped — simulates process exit

        // ── Process B: fresh persistor + fresh registry; hydrate ──
        let mut registry: LcpRegistry<HybridKvCacheSnapshot> = LcpRegistry::new(8);
        let persistor_b = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let triples = persistor_b.hydrate_for_cfg(&cfg, &device).unwrap();
        assert_eq!(triples.len(), 1, "hydrate must find the one TQ snapshot");

        for (_key_hex, hydrated_snap, sidecar) in triples {
            // Byte-equality check: every TQ buffer in the hydrated snapshot
            // matches what process A wrote.
            assert_eq!(
                hydrated_snap.full_attn_tq.len(),
                n_full_attn,
                "hydrated snapshot must have {n_full_attn} full_attn_tq entries"
            );
            for (i, hyd_tq) in hydrated_snap.full_attn_tq.iter().enumerate() {
                let hyd = hyd_tq.as_ref().unwrap_or_else(|| {
                    panic!("hydrated full_attn_tq[{i}] must be Some — codec v3 should round-trip TQ payload")
                });
                assert_eq!(
                    hyd.k_packed.as_slice::<u8>().unwrap(),
                    src_k_packed[i].as_slice(),
                    "slot[{i}].k_packed bytes diverge across cross-process round-trip"
                );
                assert_eq!(
                    hyd.v_packed.as_slice::<u8>().unwrap(),
                    src_v_packed[i].as_slice(),
                    "slot[{i}].v_packed bytes diverge"
                );
                assert_eq!(
                    hyd.k_norms.as_slice::<u8>().unwrap(),
                    src_k_norms[i].as_slice(),
                    "slot[{i}].k_norms bytes diverge"
                );
                assert_eq!(
                    hyd.v_norms.as_slice::<u8>().unwrap(),
                    src_v_norms[i].as_slice(),
                    "slot[{i}].v_norms bytes diverge"
                );
                assert_eq!(hyd.norms_per_pos, 1, "norms_per_pos must round-trip");
            }
            // K/V remain None (TQ-only mode).
            for i in 0..n_full_attn {
                assert!(
                    hydrated_snap.full_attn_k[i].is_none(),
                    "K must remain None across round-trip for slot[{i}]"
                );
                assert!(
                    hydrated_snap.full_attn_v[i].is_none(),
                    "V must remain None across round-trip for slot[{i}]"
                );
            }
            // Re-store into registry (mirror production hydrate path).
            let key = LcpKey {
                model_fingerprint: sidecar.model_fingerprint.clone(),
                tenant_id: sidecar.tenant_id.clone(),
                params_hash: sidecar.params_hash,
            };
            registry
                .store(
                    key,
                    sidecar.prompt_tokens.clone(),
                    vec![std::sync::Arc::new(hydrated_snap)],
                    sidecar.sliding_window as usize,
                    sidecar.linear_capacity as usize,
                )
                .expect("registry stores hydrated TQ snapshot");
        }

        // Lookup with shared prefix (LCP=7) succeeds.
        let prefix = registry
            .lookup(&original_key, &new_request_prompt)
            .expect("post-hydrate lookup with original key must hit");
        assert_eq!(prefix.k, 7, "LCP length must match shared prefix");

        let _ = fs::remove_dir_all(&tempdir);
    }

    /// Defensive: a registry hydrated from a disk persistor with an
    /// EMPTY cfg-subdir (clean cold start on a brand-new HF2Q_KV_PERSIST
    /// dir) stays empty and lookups miss cleanly. No I/O errors, no
    /// partial state, no false-positive hits.
    #[test]
    fn qh35_disk_clean_cold_start_yields_empty_hydrate() {
        use crate::serve::kv_persist::lcp_registry::{LcpKey, LcpRegistry};
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("clean-cold");
        let _ = fs::remove_dir_all(&tempdir);

        let persistor = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let triples = persistor.hydrate_for_cfg(&cfg, &device).unwrap();
        assert!(triples.is_empty(), "fresh cache_dir must hydrate to []");

        let registry: LcpRegistry<HybridKvCacheSnapshot> = LcpRegistry::new(8);
        assert_eq!(registry.len(), 0);

        let key = LcpKey {
            model_fingerprint: ModelFingerprint([0u8; 32]),
            tenant_id: "default".to_string(),
            params_hash: 0,
        };
        let mut reg = registry;
        let lookup = reg.lookup(&key, &[1, 2, 3]);
        assert!(lookup.is_none(), "empty registry must miss every lookup");

        let _ = fs::remove_dir_all(&tempdir);
    }

    /// ADR-027 sub-iter 23d-γ — on-disk budget enforcement: oldest-mtime
    /// `.bin` files are evicted LRU-first until the cfg subdir totals ≤
    /// budget; the just-written file is never evicted; hydrate tolerates
    /// the gaps (walks existing files only).
    #[test]
    fn qh35_disk_budget_evicts_oldest_keeps_newest() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = synth_cfg();
        let tempdir = unique_tempdir("budget");
        let _ = fs::remove_dir_all(&tempdir);

        // Measure one file's on-disk size, then set the budget to hold
        // exactly two.
        let probe = Qwen35DiskPersistor::new(tempdir.clone()).unwrap();
        let snap = synth_snapshot(&device, &cfg);
        let sidecar = synth_sidecar();
        probe.write(&cfg, "probe_key", &snap, &sidecar).unwrap();
        let one_size = fs::metadata(probe.path_for_key(&cfg, "probe_key"))
            .unwrap()
            .len();
        let _ = fs::remove_dir_all(&tempdir);

        let persistor =
            Qwen35DiskPersistor::new_with_budget(tempdir.clone(), one_size * 2).unwrap();
        for key in ["k_oldest", "k_middle", "k_newest"] {
            persistor.write(&cfg, key, &snap, &sidecar).unwrap();
            // Distinct mtimes for deterministic LRU order.
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        assert!(
            !persistor.path_for_key(&cfg, "k_oldest").exists(),
            "oldest snapshot must be evicted under budget"
        );
        assert!(
            persistor.path_for_key(&cfg, "k_newest").exists(),
            "newest snapshot must always survive"
        );
        // Subdir total ≤ budget (the just-written keep is exempt by
        // design, but here 2 × one_size == budget so middle survives too).
        let dir = tempdir.join(
            crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::fingerprint_hex_for(&cfg),
        );
        let total: u64 = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        assert!(
            total <= one_size * 2,
            "subdir total {total} exceeds budget {}",
            one_size * 2
        );

        // Hydrate tolerates the evicted gap: it returns exactly the
        // surviving entries, no error.
        let triples = persistor.hydrate_for_cfg(&cfg, &device).unwrap();
        assert_eq!(
            triples.len(),
            2,
            "hydrate must return the two surviving snapshots"
        );

        let _ = fs::remove_dir_all(&tempdir);
    }
}
