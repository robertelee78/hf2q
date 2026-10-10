//! ADR-062 D4 — DeepSeek-V4 recovery-anchor disk persistence.
//!
//! The Qwen 3.5 precedent (`qwen35_disk_persistor`) persists chunk-level
//! LCP snapshots keyed by `LcpKey`; the DeepSeek-V4 recovery anchor is
//! conversation-shaped instead, so this persistor keeps ONE prefix
//! image per conversation: the exact recovery-anchor state (circular
//! window rows at the anchor position, recurrent compressor pools, and
//! the append-only compressed/indexer rows below the anchor) serialized
//! by `Deepseek4Cache::serialize_anchor_image`.
//!
//! ## File-on-disk contract
//!
//! ```text
//! <cache_dir>/ds4-<model-fingerprint>/<key>.dsimg
//! ```
//!
//! `<model-fingerprint>` is a 16-hex-char digest of the model's
//! cache-shape identity (layer count, head dims, compression schedule,
//! sliding window, KV head count) plus the quant label, so an image
//! written by one artifact never hydrates against another. `<key>` is
//! the SHA-256 prefix of the anchor's exact rendered token ledger.
//! Atomic writes use tempfile + rename; a trailing SHA-256 in the file
//! catches corruption at hydrate time.
//!
//! ## Matching (no state leaks between conversations)
//!
//! Hydration is only offered when an image's token ledger is an EXACT
//! proper prefix of the rendered prompt. KV state is a pure function of
//! (model artifact, token prefix), so an exact-prefix match cannot leak
//! another conversation's state — two conversations that share a prefix
//! share its KV bytes by construction. Non-prefix conversations never
//! match. When a conversation grows, its new anchor image supersedes
//! (deletes) the strictly-shorter prefix images of the same
//! conversation, keeping the one-image-per-conversation invariant and
//! bounding disk growth; the byte budget (`--kv-persist-budget`) evicts
//! oldest-mtime images beyond it.
//!
//! ## Scheduler coverage
//!
//! The engine hooks live in the shared prefill/commit seams
//! (`Deepseek4LoadedModel::prefill_suffix`,
//! `begin_resumable_cold_prefill`, and both `commit_request_anchor`
//! impls — the loaded model's and each agent slot session's), so both
//! schedulers (SerialFifo and SlotAware `inflight-batched`) persist and
//! hydrate.
//!
//! ## Failure posture
//!
//! A missing, corrupt, or incompatible image only causes a cold prefill
//! (warn, never fatal). Corrupt files are best-effort deleted so the
//! next start does not retry them. Persistence-layer write failure is
//! warn-and-continue — the in-memory serving path is unaffected.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::inference::models::deepseek4::cache::{
    AnchorImageHeader, Deepseek4Cache, Deepseek4CacheSnapshot,
};
use crate::inference::models::deepseek4::Deepseek4Config;
use crate::serve::kv_persist::format::BLOCK_TOKENS;
use crate::serve::kv_persist::{EngineBindable, KvCacheSpill};
use crate::serve::multi_model::SpillErrorKind;

/// Images below this token count are never written or hydrated: the
/// engine's matrix-prefill reuse policy resets on such short prefixes
/// anyway (`prefer_matrix_prefill` requires a >=128-token cached prefix
/// for a non-trivial suffix), so persisting them would only burn disk.
pub const MIN_HYDRATABLE_PREFIX_TOKENS: usize = 128;

/// ADR-062 D4 family hook. The anchor persistor below owns the real
/// persistence (driven from the engine's commit/prefill seams through
/// `LoadOptions::kv_persist_dir`), NOT the spiller's
/// `(layer, range)` block contract — DeepSeek's compressed/window/
/// recurrent layout does not fit the block shape, so this hook mirrors
/// `StubGemma4Spill` semantics (snapshot `None` → `Skipped`). Registering
/// it at `cmd_serve` makes `KvSpiller::family_persist_active` report
/// true for a DeepSeek-V4 `(repo, quant)` — honest now that the engine
/// actually persists and hydrates.
#[derive(Debug, Default)]
pub struct Deepseek4AnchorSpill;

impl KvCacheSpill for Deepseek4AnchorSpill {
    fn block_alignment(&self) -> u32 {
        BLOCK_TOKENS
    }
    fn snapshot_block(
        &self,
        _layer_rank: usize,
        _range: std::ops::Range<u32>,
    ) -> Option<Vec<u8>> {
        // The anchor image substrate does not use block snapshots.
        None
    }
    fn restore_block(
        &mut self,
        _layer_rank: usize,
        _range: std::ops::Range<u32>,
        _payload: &[u8],
    ) -> Result<(), SpillErrorKind> {
        Err(SpillErrorKind::CodecErr)
    }
}

impl EngineBindable for Deepseek4AnchorSpill {
    fn bind_engine(&self, _engine_dyn: Arc<dyn std::any::Any + Send + Sync>) {
        // No-op: the anchor persistor is wired engine-side at load.
    }
    fn unbind_engine(&self) {
        // No-op.
    }
}

/// One in-memory index entry for a persisted image (header facts only —
/// the payload stays on disk until a hydrate reads it).
#[derive(Clone, Debug)]
struct ImageEntry {
    key_hex: String,
    tokens: Vec<u32>,
    anchor_position: usize,
}

#[derive(Debug)]
struct WriteJob {
    key_hex: String,
    tokens: Vec<u32>,
    anchor_position: usize,
    bytes: Arc<Vec<u8>>,
}

#[derive(Default)]
struct WriterState {
    shutdown: bool,
    pending: Option<WriteJob>,
    busy: bool,
}

struct AsyncWriter {
    shared: Arc<(Mutex<WriterState>, Condvar)>,
    handle: Option<JoinHandle<()>>,
}

impl AsyncWriter {
    fn spawn(
        model_dir: PathBuf,
        budget_bytes: u64,
        index: Arc<Mutex<Vec<ImageEntry>>>,
    ) -> Self {
        let shared = Arc::new((Mutex::new(WriterState::default()), Condvar::new()));
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("hf2q-ds4-anchor-persist".into())
            .spawn(move || {
                let (lock, idle) = &*worker_shared;
                loop {
                    let job = {
                        let mut state = lock
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        while state.pending.is_none() && !state.shutdown {
                            state = idle
                                .wait(state)
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                        }
                        match state.pending.take() {
                            Some(job) => {
                                state.busy = true;
                                job
                            }
                            None => break, // shutdown with nothing pending
                        }
                    };
                    write_job(&model_dir, budget_bytes, &index, job);
                    let mut state = lock
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    state.busy = false;
                    idle.notify_all();
                }
            })
            .expect("spawn DeepSeek-V4 anchor persistor writer");
        Self {
            shared,
            handle: Some(handle),
        }
    }

    /// Latest-wins single pending slot. Returns `Ok(true)` when an older
    /// pending job was replaced. Errors only once shutdown began.
    fn enqueue(&self, job: WriteJob) -> Result<bool> {
        let (lock, idle) = &*self.shared;
        let mut state = lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.shutdown {
            anyhow::bail!("DeepSeek-V4 anchor persistor writer is shutting down");
        }
        let replaced = state.pending.is_some();
        state.pending = Some(job);
        idle.notify_all();
        Ok(replaced)
    }

    fn pending_jobs(&self) -> usize {
        let (lock, _) = &*self.shared;
        let state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        usize::from(state.pending.is_some()) + usize::from(state.busy)
    }
}

impl Drop for AsyncWriter {
    fn drop(&mut self) {
        {
            let (lock, idle) = &*self.shared;
            let mut state = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            state.shutdown = true;
            idle.notify_all();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Disk-backed recovery-anchor persistor for DeepSeek-V4 (one image per
/// conversation). Constructed per loaded model by
/// `Deepseek4LoadedModel::load_with_context` when the typed
/// `--kv-persist` path is set; shared with every agent slot session.
pub struct Deepseek4DiskPersistor {
    model_dir: PathBuf,
    budget_bytes: u64,
    index: Arc<Mutex<Vec<ImageEntry>>>,
    writer: AsyncWriter,
}

impl Deepseek4DiskPersistor {
    /// Construct the persistor under `<cache_dir>/ds4-<fingerprint>/`,
    /// scan existing image headers into the hydrate index, and start the
    /// background writer. Fails when the directory cannot be created or
    /// scanned — the caller (engine load) turns that into a clear
    /// startup error instead of silently serving without persistence.
    pub fn new(
        cache_dir: &Path,
        cfg: &Deepseek4Config,
        quant_label: Option<&str>,
        budget_bytes: u64,
    ) -> Result<Self> {
        let model_dir = cache_dir.join(format!("ds4-{}", model_fingerprint_hex(cfg, quant_label)));
        fs::create_dir_all(&model_dir).with_context(|| {
            format!(
                "DeepSeek-V4 --kv-persist: create anchor-image dir {}",
                model_dir.display()
            )
        })?;
        let index = Arc::new(Mutex::new(Vec::new()));
        scan_model_dir(&model_dir, &index);
        let writer = AsyncWriter::spawn(model_dir.clone(), budget_bytes, Arc::clone(&index));
        Ok(Self {
            model_dir,
            budget_bytes,
            index,
            writer,
        })
    }

    /// Stable per-artifact fingerprint digest (16 hex chars): everything
    /// that shapes the persisted KV rows.
    pub fn fingerprint_hex_for(cfg: &Deepseek4Config, quant_label: Option<&str>) -> String {
        model_fingerprint_hex(cfg, quant_label)
    }

    /// Image key for an exact rendered token ledger: the SHA-256 prefix
    /// of the token bytes.
    pub fn key_hex_for(tokens: &[u32]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"DSK4-anchor-key-v1");
        for &token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        hex::encode(&hasher.finalize()[..8])
    }

    /// Longest persisted image whose token ledger is an exact proper
    /// prefix of `prompt_tokens` (and is long enough for the engine's
    /// matrix reuse policy). Returns `(key_hex, anchor_tokens)`.
    pub fn find_hydratable(&self, prompt_tokens: &[u32]) -> Option<(String, usize)> {
        let index = self.index.lock().unwrap_or_else(|p| p.into_inner());
        let mut best: Option<&ImageEntry> = None;
        for entry in index.iter() {
            if entry.tokens.len() < MIN_HYDRATABLE_PREFIX_TOKENS
                || entry.anchor_position != entry.tokens.len()
                || entry.tokens.len() >= prompt_tokens.len()
                || !prompt_tokens.starts_with(&entry.tokens)
            {
                continue;
            }
            if best.is_none_or(|current| entry.tokens.len() > current.tokens.len()) {
                best = Some(entry);
            }
        }
        best.map(|entry| (entry.key_hex.clone(), entry.tokens.len()))
    }

    /// Read one image file whole. `Ok(None)` is a clean miss (the file
    /// vanished); `Err` is an I/O failure the caller warns on.
    pub fn read_image(&self, key_hex: &str) -> Result<Option<Vec<u8>>> {
        let path = self.image_path(key_hex);
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(anyhow::anyhow!(
                "DeepSeek-V4 anchor image read {}: {error}",
                path.display()
            )),
        }
    }

    /// Report a failed hydrate. `remove_file` deletes the image so the
    /// next start does not retry it (used for corrupt files; an
    /// incompatible-but-intact image is kept for a matching future
    /// process).
    pub fn note_failed_hydrate(&self, key_hex: &str, error: &str, remove_file: bool) {
        tracing::warn!(
            target: "hf2q::serve::api::engine_deepseek4::persist",
            key = %key_hex,
            error,
            "ADR-062 D4: DeepSeek-V4 persisted anchor image unusable; cold prefill"
        );
        if remove_file {
            let path = self.image_path(key_hex);
            if let Err(remove_error) = fs::remove_file(&path) {
                if remove_error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        path = %path.display(),
                        error = %remove_error,
                        "ADR-062 D4: failed to remove unusable anchor image"
                    );
                }
            }
            let mut index = self.index.lock().unwrap_or_else(|p| p.into_inner());
            index.retain(|entry| entry.key_hex != key_hex);
        }
    }

    /// Serialize the exact recovery-anchor image from the live cache and
    /// the committed anchor snapshot, then hand it to the background
    /// writer. The GPU readback happens synchronously (the append-only
    /// rows below the anchor must be captured before any later request
    /// can overwrite them via a reset + cold prefill); the file I/O is
    /// off the request path. Any failure is warn-and-continue.
    pub fn capture_and_submit(
        &self,
        cache: &Deepseek4Cache,
        anchor: &Deepseek4CacheSnapshot,
        tokens: &[u32],
    ) {
        if tokens.len() < MIN_HYDRATABLE_PREFIX_TOKENS {
            return;
        }
        let started = Instant::now();
        let bytes = match cache.serialize_anchor_image(anchor, tokens) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(
                    target: "hf2q::serve::api::engine_deepseek4::persist",
                    error = %error,
                    "ADR-062 D4: anchor image serialize failed; skipping disk write"
                );
                return;
            }
        };
        let key_hex = Self::key_hex_for(tokens);
        let serialize_ms = started.elapsed().as_secs_f64() * 1000.0;
        let job = WriteJob {
            key_hex: key_hex.clone(),
            tokens: tokens.to_vec(),
            anchor_position: anchor.position(),
            bytes: Arc::new(bytes),
        };
        match self.writer.enqueue(job) {
            Ok(replaced_pending) => {
                tracing::info!(
                    target: "hf2q::serve::api::engine_deepseek4::persist",
                    key = %key_hex,
                    tokens = tokens.len(),
                    replaced_pending,
                    pending_writes = self.pending_writes(),
                    serialize_ms,
                    "ADR-062 D4: DeepSeek-V4 recovery-anchor image queued for disk persistence"
                );
            }
            Err(error) => {
                tracing::warn!(
                    target: "hf2q::serve::api::engine_deepseek4::persist",
                    error = %format!("{error:#}"),
                    "ADR-062 D4: anchor image enqueue failed; in-memory anchor remains live"
                );
            }
        }
    }

    pub fn pending_writes(&self) -> usize {
        self.writer.pending_jobs()
    }

    fn image_path(&self, key_hex: &str) -> PathBuf {
        self.model_dir.join(format!("{key_hex}.dsimg"))
    }
}

/// Background writer body: atomic write + index upsert + supersede +
/// budget eviction.
fn write_job(
    model_dir: &Path,
    budget_bytes: u64,
    index: &Arc<Mutex<Vec<ImageEntry>>>,
    job: WriteJob,
) {
    let final_path = model_dir.join(format!("{}.dsimg", job.key_hex));
    let tmp_path = model_dir.join(format!("{}.dsimg.tmp", job.key_hex));
    let write = (|| -> Result<()> {
        let mut tmp = fs::File::create(&tmp_path).with_context(|| {
            format!("create anchor image tmp {}", tmp_path.display())
        })?;
        tmp.write_all(&job.bytes).with_context(|| {
            format!("write anchor image {}", tmp_path.display())
        })?;
        tmp.sync_all()
            .with_context(|| format!("sync anchor image {}", tmp_path.display()))?;
        drop(tmp);
        fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "rename anchor image {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        Ok(())
    })();
    if let Err(error) = write {
        let _ = fs::remove_file(&tmp_path);
        tracing::warn!(
            target: "hf2q::serve::api::engine_deepseek4::persist",
            key = %job.key_hex,
            error = %format!("{error:#}"),
            "ADR-062 D4: anchor image write failed; continuing"
        );
        return;
    }
    let mut index = index.lock().unwrap_or_else(|p| p.into_inner());
    // One image per conversation: this write replaces any prior image
    // at the same key, and supersedes this conversation's older,
    // strictly shorter prefix images (a longer image serves every
    // prompt the shorter one could) — delete those.
    let mut superseded: Vec<String> = Vec::new();
    index.retain(|entry| {
        if entry.key_hex == job.key_hex {
            return false;
        }
        let is_superseded =
            entry.tokens.len() < job.tokens.len() && job.tokens.starts_with(&entry.tokens);
        if is_superseded {
            superseded.push(entry.key_hex.clone());
        }
        !is_superseded
    });
    index.push(ImageEntry {
        key_hex: job.key_hex.clone(),
        tokens: job.tokens.clone(),
        anchor_position: job.anchor_position,
    });
    for key_hex in superseded {
        let _ = fs::remove_file(model_dir.join(format!("{key_hex}.dsimg")));
    }
    enforce_budget(model_dir, budget_bytes, &final_path, &mut index);
}

/// LRU-by-mtime budget enforcement over `.dsimg` files (0 = unlimited).
/// The just-written file is never evicted. Index entries whose files
/// were evicted are dropped.
fn enforce_budget(
    model_dir: &Path,
    budget_bytes: u64,
    keep: &Path,
    index: &mut Vec<ImageEntry>,
) {
    if budget_bytes == 0 {
        return;
    }
    let entries = match fs::read_dir(model_dir) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                dir = %model_dir.display(),
                error = %error,
                "ADR-062 D4: anchor image budget scan failed; skipping eviction"
            );
            return;
        }
    };
    let mut files: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("dsimg") {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        files.push((path, mtime, meta.len()));
    }
    let mut total: u64 = files.iter().map(|file| file.2).sum();
    if total <= budget_bytes {
        return;
    }
    files.sort_by_key(|file| file.1);
    for (path, _, size) in files {
        if total <= budget_bytes {
            break;
        }
        if path == keep {
            continue;
        }
        if fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(size);
            if let Some(key_hex) = path.file_stem().and_then(|s| s.to_str()) {
                index.retain(|entry| entry.key_hex != key_hex);
            }
        }
    }
}

/// Startup scan: read each image's header (fixed header + token ledger
/// only — never the payload) into the hydrate index. Corrupt headers
/// are warn-skipped; the file stays on disk where the budget or the
/// operator can clean it up.
fn scan_model_dir(model_dir: &Path, index: &Arc<Mutex<Vec<ImageEntry>>>) {
    let entries = match fs::read_dir(model_dir) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                dir = %model_dir.display(),
                error = %error,
                "ADR-062 D4: anchor image scan failed; hydrate index starts empty"
            );
            return;
        }
    };
    let mut scanned = 0usize;
    let mut skipped = 0usize;
    let mut entries_out = index.lock().unwrap_or_else(|p| p.into_inner());
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("dsimg") {
            continue;
        }
        let Some(key_hex) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        match scan_image_header(&path) {
            Ok((header, tokens)) => {
                entries_out.push(ImageEntry {
                    key_hex: key_hex.to_string(),
                    tokens,
                    anchor_position: header.anchor_position,
                });
                scanned += 1;
            }
            Err(error) => {
                skipped += 1;
                tracing::warn!(
                    path = %path.display(),
                    error = %format!("{error:#}"),
                    "ADR-062 D4: skipping unreadable anchor image header"
                );
            }
        }
    }
    tracing::info!(
        dir = %model_dir.display(),
        scanned,
        skipped,
        "ADR-062 D4: DeepSeek-V4 anchor image scan complete"
    );
}

/// Stable 16-hex-char compatibility fingerprint of everything that
/// shapes the persisted KV rows: the compression schedule, head dims,
/// window, KV head count, layer count, and the quant label. An image
/// written by one artifact lands in its own `ds4-<fingerprint>`
/// subdir and can never hydrate against a differently-shaped model.
fn model_fingerprint_hex(cfg: &Deepseek4Config, quant_label: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"DSK4-anchor-fp-v1");
    hasher.update(&cfg.num_hidden_layers.to_le_bytes());
    hasher.update(&cfg.num_key_value_heads.to_le_bytes());
    hasher.update(&cfg.head_dim.to_le_bytes());
    hasher.update(&cfg.index_head_dim.to_le_bytes());
    hasher.update(&cfg.sliding_window.to_le_bytes());
    hasher.update(&(cfg.compress_ratios.len() as u32).to_le_bytes());
    for &ratio in &cfg.compress_ratios {
        hasher.update(&ratio.to_le_bytes());
    }
    hasher.update(quant_label.unwrap_or("unknown").as_bytes());
    hex::encode(&hasher.finalize()[..8])
}

/// Partial-read one image file: fixed header, then the token ledger —
/// never the payload. A short/corrupt file errors and is warn-skipped.
fn scan_image_header(path: &Path) -> Result<(AnchorImageHeader, Vec<u32>)> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("open anchor image header {}", path.display()))?;
    let mut fixed = [0u8; 40];
    file.read_exact(&mut fixed)
        .with_context(|| format!("read anchor image header {}", path.display()))?;
    let (header, _) = AnchorImageHeader::parse_prefix(&fixed)
        .with_context(|| format!("parse anchor image header {}", path.display()))?;
    let mut token_bytes = vec![0u8; header.token_count * 4];
    file.read_exact(&mut token_bytes)
        .with_context(|| format!("read anchor image tokens {}", path.display()))?;
    let mut prefix = fixed.to_vec();
    prefix.extend_from_slice(&token_bytes);
    let (_, tokens) = AnchorImageHeader::parse_prefix(&prefix)
        .with_context(|| format!("reparse anchor image prefix {}", path.display()))?;
    let tokens =
        tokens.ok_or_else(|| anyhow::anyhow!("token ledger missing after partial read"))?;
    Ok((header, tokens))
}
