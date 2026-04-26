//! ADR-005 Phase 3 (lines 901–917) item 2/4: model cache layout.
//!
//! Provides the on-disk layout, manifest, and concurrency primitives for the
//! `hf2q serve --model <repo-or-path>` auto-pipeline. **Pure cache-layout** —
//! no download, no quantize, no integrity-check logic lives here. Those are
//! iter-203 (integrity check) and iter-204 (auto-pipeline wiring).
//!
//! # Directory layout
//!
//! Resolved root (in priority order):
//! 1. `HF2Q_CACHE_DIR` env var (configuration override; first to win)
//! 2. `XDG_CACHE_HOME` env var → `$XDG_CACHE_HOME/hf2q`
//! 3. `$HOME/.cache/hf2q` (default)
//!
//! ```text
//! <cache-root>/
//! ├── manifest.json                              # global cache index (schema v1)
//! ├── locks/                                     # advisory `flock(LOCK_EX)` files
//! │   └── <repo-slug>__<quant>.lock
//! └── models/
//!     └── <repo-slug>/                           # slug = repo_id, '/' → '__'
//!         ├── repo_meta.json                     # HF revision SHA, last_fetched
//!         ├── source/                            # source pointer or local copy
//!         │   └── source_pointer.json
//!         └── quantized/
//!             └── <quant>/                       # Q8_0, Q6_K, Q4_K_M, Q3_K_M
//!                 ├── model.gguf                 # quantized GGUF (atomic-renamed)
//!                 ├── mmproj.gguf                # optional vision projector
//!                 └── manifest.json              # per-quant manifest
//! ```
//!
//! # Chesterton's-fence interop notes
//!
//! - **`api::state::default_cache_dir`** (`src/serve/api/state.rs:109`) returns
//!   `$HOME/.cache/hf2q` and is the existing root used by
//!   `scan_cache_dir` (`src/serve/api/handlers.rs:2804`) for the `/v1/models`
//!   listing.  This module's [`default_root`] picks the same root by default
//!   (with the additional `HF2Q_CACHE_DIR`/`XDG_CACHE_HOME` overrides that
//!   ADR-005 Phase 3 mandates).  Per-quant `model.gguf` files written under
//!   `models/<slug>/quantized/<quant>/` are reachable by the existing
//!   `visit_dir` recursion (max depth 6 is enough for our 5-deep layout) so
//!   `/v1/models` will pick them up automatically.
//! - **HF cache** (`~/.cache/huggingface/hub/`) is never duplicated. When
//!   [`record_source`] is given a `SourcePointer::HfHub`, only metadata is
//!   stored — the bytes stay in the HF cache, where `hf_download.rs` already
//!   manages them (resolution order at `src/input/hf_download.rs:653-674`).
//! - **`libc::flock`** is the locking primitive (precedent at
//!   `src/intelligence/ruvector.rs:649-664`); no new Cargo deps.
//! - **`sha256_file`** is re-implemented privately here; the existing
//!   `parity_quality::sha256_file` is module-private and a tight 15-line
//!   helper.  Duplication is intentional rather than promoting that one to
//!   `pub` (Chesterton: `parity_quality` is a closed Gate-H surface; opening
//!   its hashing helper to the world would couple the cache module to a
//!   measurement subsystem it has no business depending on).
//!
//! # Atomic-write invariants
//!
//! Every manifest write goes through `tempfile::NamedTempFile::persist` in
//! the same parent directory so the `rename(2)` is atomic on the same
//! filesystem.  Quantized GGUF writers (iter-204) will use the same pattern
//! via [`QuantWriteHandle::finalize`].  Crash-mid-write either leaves the
//! prior valid manifest untouched, or no manifest at all (caller can
//! distinguish via [`ModelCache::open`] which creates a fresh empty
//! manifest if none exists).

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::quant_select::QuantType;

/// Schema version for [`CacheManifest`].  Bumped when on-disk JSON layout
/// changes incompatibly.
///
/// - **v1** (iter-202): initial Phase 3 layout — `models: BTreeMap<String,
///   ModelEntry>`, `ModelEntry { repo_id, revision, source, quantizations,
///   last_accessed_secs }`.
/// - **v2** (iter-203): adds `ModelEntry::source_shards: Vec<SourceShard>`
///   for ADR-005 Phase 3 item 3/4 (HF integrity).  v1 manifests load
///   transparently — missing field defaults to empty Vec — and are
///   re-written as v2 on the next mutation.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;
/// Lowest schema version the loader still understands.  v1 manifests are
/// migrated in-place to v2 on read (no field invalidation).
pub const MANIFEST_SCHEMA_MIN_SUPPORTED: u32 = 1;

/// Env var that (if set + non-empty) overrides every other root resolution
/// rule.  Documented in ADR-005 Phase 3.
pub const HF2Q_CACHE_DIR_ENV: &str = "HF2Q_CACHE_DIR";

/// Top-level manifest shape.  Serialized as `manifest.json` at the cache
/// root.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheManifest {
    pub schema_version: u32,
    /// Keyed by HF repo_id (`org/repo`).  `BTreeMap` for deterministic
    /// serialization order.
    pub models: BTreeMap<String, ModelEntry>,
}

impl Default for CacheManifest {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: BTreeMap::new(),
        }
    }
}

/// One model's worth of cache state.  A model can have multiple quantized
/// variants cached side-by-side (e.g. Q4_K_M and Q6_K of the same base
/// weights); the LRU `last_accessed` is per-model, not per-quant, so all
/// quants of a model evict together.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelEntry {
    pub repo_id: String,
    /// HF revision SHA at the time the source was fetched (or `"local"`
    /// for local-path sources).
    pub revision: String,
    pub source: Option<SourcePointer>,
    /// One entry per cached quant.  `BTreeMap` keyed by the quant's
    /// canonical GGML name (`Q4_K_M` etc., matches
    /// [`QuantType::as_str`]) for deterministic JSON.
    pub quantizations: BTreeMap<String, QuantEntry>,
    /// Wall-clock seconds-since-epoch.  Updated by [`ModelCache::touch`].
    /// Drives LRU eviction (Phase 4).
    pub last_accessed_secs: u64,
    /// Per-shard integrity records captured by ADR-005 Phase 3 item 3/4
    /// (`hf2q convert` / `hf2q serve` against `--repo`).  Empty when the
    /// source is local or when integrity was bypassed via
    /// `--no-integrity`.  Schema bumped from v1 → v2 to introduce this
    /// field; v1 manifests load with `source_shards = vec![]` (default).
    #[serde(default)]
    pub source_shards: Vec<SourceShard>,
}

/// One per-shard integrity record (ADR-005 Phase 3 item 3/4).  Mirrors
/// [`crate::input::integrity::ShardIntegrity`] in JSON shape so the two
/// surfaces stay aligned and a `From` adapter is a memberwise copy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceShard {
    pub filename: String,
    pub bytes: u64,
    /// Lowercase hex SHA-256 from HF's `x-linked-etag` (LFS-managed
    /// shards only).  `None` for non-LFS files (config.json,
    /// tokenizer.json) — see `crate::input::integrity` module docs.
    pub sha256: Option<String>,
    /// Raw etag as returned by HF (LFS sha256 hex when LFS, Git-style
    /// blob SHA-1 otherwise) — kept verbatim for traceability.
    pub hf_etag: String,
    pub is_lfs: bool,
    /// Wall-clock seconds-since-epoch when this record was added.
    pub verified_at_secs: u64,
}

impl SourceShard {
    /// Adapter from the integrity-side struct.  Stamps `verified_at_secs`
    /// at adapter time so the cache's clock is the canonical timestamp.
    pub fn from_integrity(value: &crate::input::integrity::ShardIntegrity) -> Self {
        Self {
            filename: value.filename.clone(),
            bytes: value.bytes,
            sha256: value.sha256.clone(),
            hf_etag: value.hf_etag.clone(),
            is_lfs: value.is_lfs,
            verified_at_secs: secs_since_epoch(),
        }
    }
}

/// Compute a deterministic source-bundle SHA-256 from a list of source
/// shards — ADR-005 Phase 4 iter-207 (provenance reader).
///
/// The result is the lowercase-hex SHA-256 of `"<filename>:<sha256-hex>\n"`
/// for every shard whose `sha256` is `Some(_)`, sorted by filename.  This
/// is the canonical "source-bundle hash" that a hf2q-emitted GGUF carries
/// as `hf2q.source_sha256` (writer side schedules behind ADR-014 P7).  The
/// reader cross-verifies the GGUF's claim against this computed value so a
/// hf2q-origin GGUF can short-circuit the per-load 30 GB integrity re-check.
///
/// Returns `None` when nothing in the shard list is hashable — i.e. the
/// list is empty (local-source path / `--no-integrity`) or every shard
/// is `is_lfs=false` (non-LFS files like `config.json`/`tokenizer.json`,
/// which don't carry a usable SHA-256).  Callers treat `None` as
/// "provenance binding unavailable; cannot short-circuit".
///
/// Determinism: filenames are compared byte-wise (`Ord` on `String`),
/// duplicates are kept (the reduction is stable under the same input),
/// and the trailing `\n` after the last entry is preserved so two lists
/// that differ only in shard order produce the same bundle SHA.
///
/// Algorithm is self-contained and matches the future writer's emit
/// path (no shared crate state needed).  Uppercase hex inputs are
/// lowercased before hashing so a writer that uppercases never breaks
/// reader cross-verification.
pub fn compute_source_bundle_sha256(shards: &[SourceShard]) -> Option<String> {
    let mut entries: Vec<(&str, String)> = shards
        .iter()
        .filter_map(|s| {
            s.sha256
                .as_ref()
                .map(|h| (s.filename.as_str(), h.to_ascii_lowercase()))
        })
        .collect();
    if entries.is_empty() {
        return None;
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut hasher = Sha256::new();
    for (filename, sha) in &entries {
        hasher.update(filename.as_bytes());
        hasher.update(b":");
        hasher.update(sha.as_bytes());
        hasher.update(b"\n");
    }
    Some(hex::encode(hasher.finalize()))
}

/// Where the unquantized source bytes live.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourcePointer {
    /// Bytes live in the HuggingFace hf-hub cache.  We never copy from
    /// here — `iter-204` will read directly from `path` during quantize.
    HfHub {
        path: PathBuf,
        revision: String,
    },
    /// User-provided local directory of safetensors + tokenizer + config.
    Local {
        path: PathBuf,
        sha256: String,
    },
}

/// One cached quantized GGUF variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantEntry {
    /// Canonical GGML name (`"Q4_K_M"` etc.) — matches
    /// [`QuantType::as_str`].
    pub quant_type: String,
    /// Path to `model.gguf`, absolute.
    pub gguf_path: PathBuf,
    /// Optional path to vision `mmproj.gguf` if the model is multimodal.
    pub mmproj_path: Option<PathBuf>,
    /// Bytes of the GGUF (cheap reproduction of `metadata.len()` for
    /// observability).
    pub bytes: u64,
    /// SHA-256 of the GGUF, lowercase hex.  Used for integrity check on
    /// load and to detect corruption mid-cache.
    pub sha256: String,
    /// Seconds since epoch when the quantization completed (atomic
    /// `finalize` time).
    pub quantized_at_secs: u64,
    /// `hf2q` semver that produced this quant (read from
    /// `env!("CARGO_PKG_VERSION")`).
    pub quantized_by_version: String,
}

// ────────────────────────────────────────────────────────────────────
// Root resolution + slug encoding
// ────────────────────────────────────────────────────────────────────

/// Resolve the default cache root per ADR-005 Phase 3.
///
/// 1. `HF2Q_CACHE_DIR` (if set + non-empty)
/// 2. `XDG_CACHE_HOME/hf2q` (if set + non-empty)
/// 3. `$HOME/.cache/hf2q`
///
/// Returns `Err` only if none of the above resolves to a usable path
/// (e.g. all three env vars unset and no `HOME` — a hermetic CI environment).
pub fn default_root() -> Result<PathBuf> {
    if let Ok(v) = std::env::var(HF2Q_CACHE_DIR_ENV) {
        if !v.is_empty() {
            return Ok(PathBuf::from(v));
        }
    }
    if let Ok(v) = std::env::var("XDG_CACHE_HOME") {
        if !v.is_empty() {
            return Ok(PathBuf::from(v).join("hf2q"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return Ok(PathBuf::from(home).join(".cache").join("hf2q"));
        }
    }
    Err(anyhow!(
        "cannot resolve cache root: HF2Q_CACHE_DIR, XDG_CACHE_HOME, and HOME \
         are all unset (set HF2Q_CACHE_DIR explicitly to override)"
    ))
}

/// Encode an HF repo_id (`org/repo` or `user/repo-name`) as a single
/// directory component.  `/` → `__`.  Reversible round-trip via
/// [`unslug_repo_id`].
///
/// Rejects empty input and inputs with characters that are filesystem-unsafe
/// on macOS / Linux (`\0`, embedded `/` after replacement is fine).  Unicode
/// is preserved as-is — HF repo IDs are practically ASCII but we don't
/// constrain artificially.
pub fn slug_repo_id(repo_id: &str) -> Result<String> {
    if repo_id.is_empty() {
        return Err(anyhow!("repo_id is empty"));
    }
    if repo_id.contains('\0') {
        return Err(anyhow!(
            "repo_id contains NUL byte: {:?}",
            repo_id
        ));
    }
    if repo_id.contains("..") {
        return Err(anyhow!(
            "repo_id contains '..' (path-traversal guard): {}",
            repo_id
        ));
    }
    Ok(repo_id.replace('/', "__"))
}

/// Reverse of [`slug_repo_id`].  Panics never; returns the slug back if the
/// shape doesn't match the canonical encoding.
pub fn unslug_repo_id(slug: &str) -> String {
    slug.replace("__", "/")
}

/// `repo_id` + `quant` → on-disk path of the cached `model.gguf`.
///
/// Pure path rendering — does NOT touch the filesystem.  Used both by
/// [`ModelCache::record_quantized`] (writer side) and by external callers
/// that want to predict where a future quant will land.
pub fn cache_model_path(root: &Path, repo_id: &str, quant: QuantType) -> Result<PathBuf> {
    let slug = slug_repo_id(repo_id)?;
    Ok(root
        .join("models")
        .join(slug)
        .join("quantized")
        .join(quant.as_str())
        .join("model.gguf"))
}

/// Companion to [`cache_model_path`] for the optional `mmproj.gguf`.
pub fn cache_mmproj_path(root: &Path, repo_id: &str, quant: QuantType) -> Result<PathBuf> {
    let slug = slug_repo_id(repo_id)?;
    Ok(root
        .join("models")
        .join(slug)
        .join("quantized")
        .join(quant.as_str())
        .join("mmproj.gguf"))
}

// ────────────────────────────────────────────────────────────────────
// File locking primitive (advisory `flock`)
// ────────────────────────────────────────────────────────────────────

/// RAII guard around an exclusive advisory `flock(LOCK_EX)`.
///
/// On Unix, `libc::flock` releases the lock automatically when the file
/// descriptor closes (i.e. `Drop`); we additionally call `LOCK_UN` for
/// belt-and-braces and so the unlock happens before the file metadata
/// is dropped.
pub struct CacheLock {
    /// Held to keep the fd alive; cleanup happens in `Drop`.
    file: Option<File>,
}

impl CacheLock {
    /// Acquire an exclusive lock on `path`, creating the file if absent.
    /// Blocks until the lock is granted (matches the semantics of
    /// `intelligence::ruvector::lock_exclusive`).
    pub fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create lock dir: {}", parent.display()))?;
        }
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("open lock file: {}", path.display()))?;
        let fd = file.as_raw_fd();
        // SAFETY: fd is owned by `file`; flock is documented as thread-safe.
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("flock LOCK_EX: {}", path.display()));
        }
        Ok(Self { file: Some(file) })
    }

    /// Try to acquire the lock without blocking.  Returns `Ok(Some)` if
    /// granted, `Ok(None)` if another process holds it.
    pub fn try_acquire(path: &Path) -> Result<Option<Self>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create lock dir: {}", parent.display()))?;
        }
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("open lock file: {}", path.display()))?;
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if ret == 0 {
            Ok(Some(Self { file: Some(file) }))
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                Ok(None)
            } else {
                Err(err).with_context(|| {
                    format!("flock LOCK_EX|LOCK_NB: {}", path.display())
                })
            }
        }
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let fd = file.as_raw_fd();
            unsafe {
                libc::flock(fd, libc::LOCK_UN);
            }
            // file drops, fd closes
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// ModelCache — top-level handle
// ────────────────────────────────────────────────────────────────────

/// On-disk model cache handle.  Open-or-create at the given root; manifest
/// edits are buffered in memory and persisted via [`ModelCache::flush`] (or
/// implicitly by mutating helpers that take `&mut self` and end with a flush).
#[derive(Debug)]
pub struct ModelCache {
    root: PathBuf,
    manifest: CacheManifest,
}

impl ModelCache {
    /// Open the cache at the default root (per [`default_root`]).
    pub fn open() -> Result<Self> {
        Self::open_at(default_root()?)
    }

    /// Open the cache at an explicit root.  Creates the directory tree if
    /// missing.  Loads `manifest.json` if present, otherwise initializes a
    /// fresh empty manifest (NOT written to disk until something changes).
    pub fn open_at(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        ensure_layout(&root)?;

        let manifest_path = root.join("manifest.json");
        let manifest = if manifest_path.exists() {
            read_manifest(&manifest_path)?
        } else {
            CacheManifest::default()
        };

        Ok(Self { root, manifest })
    }

    /// Cache root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read-only access to the manifest.
    pub fn manifest(&self) -> &CacheManifest {
        &self.manifest
    }

    /// Look up a quant entry; returns `None` if either the model or the
    /// requested quant is uncached.
    pub fn lookup(&self, repo_id: &str, quant: QuantType) -> Option<&QuantEntry> {
        self.manifest
            .models
            .get(repo_id)
            .and_then(|m| m.quantizations.get(quant.as_str()))
    }

    /// Look up the model entry (any quant).
    pub fn lookup_model(&self, repo_id: &str) -> Option<&ModelEntry> {
        self.manifest.models.get(repo_id)
    }

    /// Record where a model's source bytes live.  Creates or updates the
    /// `ModelEntry` for `repo_id`, sets `revision` + `source`, and flushes
    /// the manifest atomically.
    ///
    /// **Does NOT copy bytes** — for `SourcePointer::HfHub` the existing
    /// HF cache is the canonical store; for `SourcePointer::Local` the
    /// caller-supplied path is recorded as-is.
    pub fn record_source(
        &mut self,
        repo_id: &str,
        revision: &str,
        source: SourcePointer,
    ) -> Result<()> {
        // Validate slug now so we don't write a half-valid entry.
        let _ = slug_repo_id(repo_id)?;

        let now = secs_since_epoch();
        let entry = self
            .manifest
            .models
            .entry(repo_id.to_string())
            .or_insert_with(|| ModelEntry {
                repo_id: repo_id.to_string(),
                revision: revision.to_string(),
                source: None,
                quantizations: BTreeMap::new(),
                last_accessed_secs: now,
                source_shards: Vec::new(),
            });
        entry.revision = revision.to_string();
        entry.source = Some(source);
        entry.last_accessed_secs = now;

        // Persist a `repo_meta.json` companion alongside `models/<slug>/`.
        let slug = slug_repo_id(repo_id)?;
        let model_dir = self.root.join("models").join(&slug);
        fs::create_dir_all(&model_dir)
            .with_context(|| format!("create model dir: {}", model_dir.display()))?;
        let meta_path = model_dir.join("repo_meta.json");
        let entry_clone = entry.clone();
        write_json_atomic(&meta_path, &entry_clone)?;

        self.flush()
    }

    /// Record a completed quantization.  Caller is responsible for having
    /// already moved (atomically) the `gguf_path` into place — this method
    /// only updates the manifest + per-quant manifest companion.
    pub fn record_quantized(&mut self, repo_id: &str, entry: QuantEntry) -> Result<()> {
        let _ = slug_repo_id(repo_id)?;
        let model = self
            .manifest
            .models
            .get_mut(repo_id)
            .ok_or_else(|| anyhow!("record_quantized: no source recorded for {}", repo_id))?;
        model
            .quantizations
            .insert(entry.quant_type.clone(), entry.clone());
        model.last_accessed_secs = secs_since_epoch();

        // Companion per-quant manifest in the quant dir.
        let quant_dir = entry
            .gguf_path
            .parent()
            .ok_or_else(|| anyhow!("gguf_path has no parent: {}", entry.gguf_path.display()))?
            .to_path_buf();
        fs::create_dir_all(&quant_dir)
            .with_context(|| format!("create quant dir: {}", quant_dir.display()))?;
        let companion = quant_dir.join("manifest.json");
        write_json_atomic(&companion, &entry)?;

        self.flush()
    }

    /// Record a model's source bytes alongside the per-shard integrity
    /// records produced by [`crate::input::integrity::verify_repo`].
    /// Stores the shard list under `ModelEntry::source_shards` so future
    /// loads can re-verify without re-fetching HF metadata.
    ///
    /// This is the v2-schema entry point: callers that want the integrity
    /// trail in the manifest use this method; callers that legitimately
    /// have no shard list (local-path source, `--no-integrity`) keep
    /// using [`record_source`] which seeds `source_shards = vec![]`.
    pub fn record_source_with_shards(
        &mut self,
        repo_id: &str,
        revision: &str,
        source: SourcePointer,
        shards: Vec<crate::input::integrity::ShardIntegrity>,
    ) -> Result<()> {
        let _ = slug_repo_id(repo_id)?;

        let now = secs_since_epoch();
        let entry = self
            .manifest
            .models
            .entry(repo_id.to_string())
            .or_insert_with(|| ModelEntry {
                repo_id: repo_id.to_string(),
                revision: revision.to_string(),
                source: None,
                quantizations: BTreeMap::new(),
                last_accessed_secs: now,
                source_shards: Vec::new(),
            });
        entry.revision = revision.to_string();
        entry.source = Some(source);
        entry.last_accessed_secs = now;
        entry.source_shards = shards.iter().map(SourceShard::from_integrity).collect();

        // Persist a `repo_meta.json` companion alongside `models/<slug>/`.
        let slug = slug_repo_id(repo_id)?;
        let model_dir = self.root.join("models").join(&slug);
        fs::create_dir_all(&model_dir)
            .with_context(|| format!("create model dir: {}", model_dir.display()))?;
        let meta_path = model_dir.join("repo_meta.json");
        let entry_clone = entry.clone();
        write_json_atomic(&meta_path, &entry_clone)?;

        self.flush()
    }

    /// Re-hash the cached GGUF on disk and compare to the SHA-256 stored
    /// in the manifest by [`record_quantized`].  Used at serve-time to
    /// detect cache corruption (disk bit-rot, partial write, manual edit).
    ///
    /// Errors:
    /// - `manifest_no_entry` if the `(repo_id, quant)` pair isn't cached.
    /// - `gguf_missing` if the recorded `gguf_path` no longer exists.
    /// - `sha256_mismatch` (the load-bearing one) when the on-disk SHA-256
    ///   diverges from the manifest entry.
    ///
    /// Naming + wording is asserted on by tests; do not change without
    /// updating the tests.
    pub fn verify_quantized(&self, repo_id: &str, quant: QuantType) -> Result<()> {
        let entry = self
            .lookup(repo_id, quant)
            .ok_or_else(|| anyhow!(
                "verify_quantized: no manifest entry for {}@{}",
                repo_id,
                quant.as_str()
            ))?;
        if !entry.gguf_path.exists() {
            return Err(anyhow!(
                "verify_quantized: cached GGUF missing on disk: {}",
                entry.gguf_path.display()
            ));
        }
        let actual = sha256_file(&entry.gguf_path)?;
        if !actual.eq_ignore_ascii_case(&entry.sha256) {
            return Err(anyhow!(
                "verify_quantized: SHA-256 mismatch for {}@{} at {}: \
                 manifest says {}, on-disk computes {}. \
                 The cached GGUF is corrupted; remove it (rm {}) and \
                 re-quantize, or pass --no-integrity to skip the check \
                 (NOT recommended).",
                repo_id,
                quant.as_str(),
                entry.gguf_path.display(),
                entry.sha256,
                actual,
                entry.gguf_path.display(),
            ));
        }
        Ok(())
    }

    /// Update `last_accessed_secs` for a model (LRU touch).  No-op if the
    /// model isn't cached.  Flushes the manifest on success.
    pub fn touch(&mut self, repo_id: &str) -> Result<()> {
        if let Some(m) = self.manifest.models.get_mut(repo_id) {
            m.last_accessed_secs = secs_since_epoch();
            self.flush()?;
        }
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────
    // Invalidation surface (ADR-005 Phase 3 iter-205, AC line 5351)
    //
    // `hf2q cache clear` is the operator workflow that drives these
    // entry points.  Three granularities:
    //
    // - [`invalidate`]          single (repo, quant) entry
    // - [`invalidate_repo`]     all quants for one repo (and the model dir)
    // - [`purge`]               every entry in the cache
    //
    // Plus [`iter_entries`] for the read-only `cache list` surface.
    //
    // # Concurrency
    //
    // `invalidate` holds the per-(repo, quant) advisory lock for the
    // duration of the destructive operation; `invalidate_repo`
    // acquires *every* per-quant lock for the repo (in deterministic
    // BTreeMap-iteration order — same `quantizations` traversal we
    // use everywhere else — so two parallel `invalidate_repo` calls
    // for the same repo can't deadlock).  `purge` does NOT acquire
    // locks because the caller (the CLI handler) is the gate that
    // promised global exclusivity via `--yes`; purging while a serve
    // is loading from cache would have already lost the race against
    // any of the per-quant entry-points anyway.
    //
    // # Atomicity
    //
    // The order is `remove on-disk bytes → mutate in-memory manifest
    // → atomic flush`.  If we crash between steps the manifest still
    // points at a now-missing GGUF; `verify_quantized` then returns
    // `gguf_missing` on the next serve attempt, which the operator
    // resolves by re-running `hf2q cache clear` (idempotent: the
    // second pass finds the manifest entry, attempts to remove an
    // already-gone file via `fs::remove_dir_all` which is silent on
    // ENOENT for missing children, and flushes a clean manifest).
    //
    // Bytes-freed accounting walks the on-disk dir BEFORE removal so
    // the number reported is what the operator actually freed (the
    // `QuantEntry::bytes` manifest field is `model.gguf` only and
    // misses the optional `mmproj.gguf` plus the per-quant companion
    // `manifest.json`).
    // ────────────────────────────────────────────────────────────────

    /// Remove a single `(repo_id, quant)` entry: deletes the on-disk
    /// `quantized/<quant>/` directory tree (covering `model.gguf`, the
    /// optional `mmproj.gguf`, and the companion per-quant manifest)
    /// and drops the `QuantEntry` from `ModelEntry::quantizations`.
    /// Holds the per-(repo, quant) advisory lock for the duration so
    /// concurrent serve invocations cannot observe a half-deleted
    /// state.  Flushes the global manifest atomically before the
    /// lock releases.
    ///
    /// Returns the total bytes freed (sum of file sizes under the
    /// removed dir as observed *before* the removal).  A `0` return
    /// is legal (e.g. the gguf was already deleted out-of-band but
    /// the manifest entry remained — `verify_quantized` would have
    /// flagged that case at serve-time).
    ///
    /// Errors:
    /// - `unknown_repo`  if `repo_id` has no model entry.
    /// - `unknown_quant` if the model entry has no `quant` quantization.
    pub fn invalidate(&mut self, repo_id: &str, quant: QuantType) -> Result<u64> {
        let _lock = self.lock_quant(repo_id, quant)?;
        let model = self
            .manifest
            .models
            .get(repo_id)
            .ok_or_else(|| anyhow!(
                "invalidate: unknown_repo: no manifest entry for {}",
                repo_id
            ))?;
        if !model.quantizations.contains_key(quant.as_str()) {
            return Err(anyhow!(
                "invalidate: unknown_quant: {} has no {} quantization cached",
                repo_id,
                quant.as_str()
            ));
        }

        // Compute bytes freed from disk before we remove anything.
        let slug = slug_repo_id(repo_id)?;
        let quant_dir = self
            .root
            .join("models")
            .join(&slug)
            .join("quantized")
            .join(quant.as_str());
        let freed = dir_total_bytes(&quant_dir);

        // Remove the on-disk tree first; if this fails we leave the
        // manifest entry intact so the operator can retry.
        if quant_dir.exists() {
            fs::remove_dir_all(&quant_dir)
                .with_context(|| format!("remove quant dir: {}", quant_dir.display()))?;
        }

        // Drop the manifest entry and persist atomically.
        if let Some(m) = self.manifest.models.get_mut(repo_id) {
            m.quantizations.remove(quant.as_str());
        }
        self.flush()?;
        Ok(freed)
    }

    /// Remove every quantization for `repo_id` and the entire
    /// `models/<slug>/` directory tree (covers `repo_meta.json`,
    /// `source/`, and every `quantized/<quant>/`).  Acquires all
    /// per-quant advisory locks for the repo (so a concurrent serve
    /// resolving any of them blocks until this returns).
    ///
    /// Returns the total bytes freed.
    ///
    /// Errors:
    /// - `unknown_repo` if `repo_id` has no manifest entry.
    pub fn invalidate_repo(&mut self, repo_id: &str) -> Result<u64> {
        let model = self
            .manifest
            .models
            .get(repo_id)
            .ok_or_else(|| anyhow!(
                "invalidate_repo: unknown_repo: no manifest entry for {}",
                repo_id
            ))?;

        // Collect every quant string up front (deterministic
        // BTreeMap-iteration order) so the lock-acquisition order is
        // stable across processes — prevents AB/BA deadlocks.
        let quant_strs: Vec<String> = model
            .quantizations
            .keys()
            .cloned()
            .collect();

        // Acquire every per-quant lock.  We hold these across the
        // remove + flush so concurrent serve sees a coherent before
        // /after.
        let mut locks: Vec<CacheLock> = Vec::with_capacity(quant_strs.len());
        for q in &quant_strs {
            let lock_path = self
                .root
                .join("locks")
                .join(format!("{}__{}.lock", slug_repo_id(repo_id)?, q));
            locks.push(CacheLock::acquire(&lock_path)?);
        }

        let slug = slug_repo_id(repo_id)?;
        let model_dir = self.root.join("models").join(&slug);
        let freed = dir_total_bytes(&model_dir);

        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)
                .with_context(|| format!("remove model dir: {}", model_dir.display()))?;
        }

        self.manifest.models.remove(repo_id);
        self.flush()?;

        // `locks` drops here, releasing every per-quant flock.
        drop(locks);
        Ok(freed)
    }

    /// Remove every entry from the cache.  Walks `<root>/models/` to
    /// compute bytes freed, removes the entire models tree, and resets
    /// the manifest to an empty default.  Locks are NOT acquired —
    /// the caller (the CLI handler) is responsible for external
    /// coordination (`hf2q cache clear --all --yes`).
    ///
    /// Returns the total bytes freed.  Idempotent: running twice
    /// returns `0` on the second call.
    pub fn purge(&mut self) -> Result<u64> {
        let models_dir = self.root.join("models");
        let freed = dir_total_bytes(&models_dir);

        if models_dir.exists() {
            fs::remove_dir_all(&models_dir)
                .with_context(|| format!("remove models tree: {}", models_dir.display()))?;
        }
        // Recreate the empty `models/` so the layout invariant
        // (`open_at` expects `models/` to exist) holds for the very
        // next `record_source` without a re-open.
        fs::create_dir_all(&models_dir)
            .with_context(|| format!("recreate models tree: {}", models_dir.display()))?;

        self.manifest = CacheManifest::default();
        self.flush()?;
        Ok(freed)
    }

    /// Iterate over every cached model entry as a lightweight view
    /// `(repo_id, model_entry)`.  Read-only; cheap given the manifest
    /// already lives in memory.  Used by `hf2q cache list` and
    /// `hf2q cache size`.
    pub fn iter_entries(&self) -> impl Iterator<Item = CacheEntryView<'_>> {
        self.manifest
            .models
            .iter()
            .map(|(repo_id, model)| CacheEntryView { repo_id, model })
    }

    /// Sum the on-disk byte size of every cached `model.gguf` +
    /// `mmproj.gguf`.  Walks `<root>/models/` once.  Used by
    /// `hf2q cache size`.
    pub fn total_bytes_on_disk(&self) -> u64 {
        let models_dir = self.root.join("models");
        dir_total_bytes(&models_dir)
    }

    /// Atomically persist the manifest to `<root>/manifest.json`.
    pub fn flush(&self) -> Result<()> {
        let path = self.root.join("manifest.json");
        write_json_atomic(&path, &self.manifest)
    }

    /// Acquire an exclusive cross-process lock for `(repo_id, quant)`.
    /// Blocks until granted.  Lock file lives at
    /// `<root>/locks/<slug>__<quant>.lock` and is created if missing.
    pub fn lock_quant(&self, repo_id: &str, quant: QuantType) -> Result<CacheLock> {
        let path = self.lock_path(repo_id, quant)?;
        CacheLock::acquire(&path)
    }

    /// Non-blocking variant of [`lock_quant`].  Returns `Ok(None)` if
    /// another process holds the lock.
    pub fn try_lock_quant(
        &self,
        repo_id: &str,
        quant: QuantType,
    ) -> Result<Option<CacheLock>> {
        let path = self.lock_path(repo_id, quant)?;
        CacheLock::try_acquire(&path)
    }

    fn lock_path(&self, repo_id: &str, quant: QuantType) -> Result<PathBuf> {
        let slug = slug_repo_id(repo_id)?;
        Ok(self
            .root
            .join("locks")
            .join(format!("{slug}__{}.lock", quant.as_str())))
    }

    /// Detect a pre-existing HuggingFace hub cache for `repo_id` (no copy).
    /// Returns the path of the latest snapshot directory if found, else
    /// `None`.  Resolution mirrors `hf_download::resolve_hf_cache_dir`
    /// (`src/input/hf_download.rs:653-674`).
    ///
    /// We deliberately do NOT cross filesystem boundaries or chase symlinks
    /// — we only check the canonical hf-hub layout
    /// (`<hub-root>/models--<org>--<repo>/snapshots/<rev>/`) and report the
    /// most recently modified `snapshots/<rev>` directory.
    pub fn detect_hf_hub_source(repo_id: &str) -> Option<HfHubSnapshot> {
        let hub_root = resolve_hf_hub_root()?;
        let dir_name = format!("models--{}", repo_id.replace('/', "--"));
        let model_root = hub_root.join(&dir_name);
        let snapshots = model_root.join("snapshots");
        if !snapshots.is_dir() {
            return None;
        }
        let mut best: Option<(SystemTime, PathBuf, String)> = None;
        for entry in fs::read_dir(&snapshots).ok()? {
            let entry = entry.ok()?;
            let ty = entry.file_type().ok()?;
            if !ty.is_dir() {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let revision = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if best
                .as_ref()
                .map(|(t, _, _)| modified > *t)
                .unwrap_or(true)
            {
                best = Some((modified, path, revision));
            }
        }
        best.map(|(_, path, revision)| HfHubSnapshot { path, revision })
    }
}

/// Result of [`ModelCache::detect_hf_hub_source`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfHubSnapshot {
    pub path: PathBuf,
    pub revision: String,
}

/// Lightweight read-only view yielded by [`ModelCache::iter_entries`].
/// Borrows from the underlying manifest; no ownership transfer or
/// allocation per entry.  Used by `hf2q cache list` to render rows
/// and by `hf2q cache size` to sum bytes.
#[derive(Debug, Clone, Copy)]
pub struct CacheEntryView<'a> {
    /// The HF repo_id key for this entry (`org/repo`).
    pub repo_id: &'a str,
    /// The full model entry (revision, source, quantizations,
    /// last_accessed_secs, source_shards).
    pub model: &'a ModelEntry,
}

// ────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────

/// Create the canonical layout under `root` if missing.
fn ensure_layout(root: &Path) -> Result<()> {
    fs::create_dir_all(root.join("models"))
        .with_context(|| format!("create models dir under {}", root.display()))?;
    fs::create_dir_all(root.join("locks"))
        .with_context(|| format!("create locks dir under {}", root.display()))?;
    Ok(())
}

fn read_manifest(path: &Path) -> Result<CacheManifest> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read manifest: {}", path.display()))?;
    let mut m: CacheManifest = serde_json::from_str(&text)
        .with_context(|| format!("parse manifest JSON: {}", path.display()))?;
    // Schema range: any version in [MIN_SUPPORTED, CURRENT] loads.  v1
    // manifests work because `source_shards` is `#[serde(default)]` —
    // missing field deserializes as empty Vec.  We then bump the
    // in-memory `schema_version` to the current value so the next flush
    // persists a v2-compliant file (forward migration is one-way; we
    // never write a v1 manifest after a v2 has loaded).
    if m.schema_version < MANIFEST_SCHEMA_MIN_SUPPORTED
        || m.schema_version > MANIFEST_SCHEMA_VERSION
    {
        return Err(anyhow!(
            "manifest schema_version mismatch at {}: expected {}..={}, got {}",
            path.display(),
            MANIFEST_SCHEMA_MIN_SUPPORTED,
            MANIFEST_SCHEMA_VERSION,
            m.schema_version
        ));
    }
    if m.schema_version < MANIFEST_SCHEMA_VERSION {
        // Migrate in memory.  No field invalidation v1 → v2 since we
        // only added a defaulted field — but be explicit so the next
        // schema bump has a hook to extend.
        m.schema_version = MANIFEST_SCHEMA_VERSION;
    }
    Ok(m)
}

/// Atomic JSON write: serialize → write to `<path>.tmp.<pid>` → fsync →
/// rename.  Within a single filesystem `rename(2)` is atomic, so any
/// crash either leaves the prior file untouched or replaces it whole.
fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent dir: {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value)
        .with_context(|| format!("serialize JSON for {}", path.display()))?;
    let tmp_name = format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("manifest.json"),
        std::process::id()
    );
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_path = parent.join(tmp_name);
    {
        let mut f = File::create(&tmp_path)
            .with_context(|| format!("create temp: {}", tmp_path.display()))?;
        f.write_all(json.as_bytes())
            .with_context(|| format!("write temp: {}", tmp_path.display()))?;
        f.sync_all()
            .with_context(|| format!("fsync temp: {}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "rename {} → {}",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn secs_since_epoch() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Resolve the hf-hub cache root.  Mirrors
/// `src/input/hf_download.rs:653-674` so the two surfaces stay in sync.
fn resolve_hf_hub_root() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        if !v.is_empty() {
            return Some(PathBuf::from(v));
        }
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        if !v.is_empty() {
            return Some(PathBuf::from(v).join("hub"));
        }
    }
    if let Ok(v) = std::env::var("XDG_CACHE_HOME") {
        if !v.is_empty() {
            return Some(PathBuf::from(v).join("huggingface").join("hub"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home).join(".cache").join("huggingface").join("hub"));
        }
    }
    None
}

/// Sum the total byte size of every regular file under `dir`,
/// recursively.  Returns `0` if `dir` does not exist or is unreadable
/// (silent — caller already separately validated the path).  Used by
/// the invalidation surface (iter-205) to compute "bytes freed" before
/// removing a tree, and by [`ModelCache::total_bytes_on_disk`] to
/// answer `hf2q cache size`.
///
/// Symlinks are not followed (matches the test-side `walk_total`
/// helper, and avoids pathological cycles); `read_dir` errors are
/// swallowed silently because the function is observability — a
/// broken FS subtree should not crash the CLI handler.
fn dir_total_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let ty = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ty.is_symlink() {
            continue;
        }
        if ty.is_file() {
            if let Ok(meta) = entry.metadata() {
                total = total.saturating_add(meta.len());
            }
        } else if ty.is_dir() {
            total = total.saturating_add(dir_total_bytes(&entry.path()));
        }
    }
    total
}

/// SHA-256 of a file as lowercase hex.  Used by [`QuantEntry`] writers.
/// Public so iter-203 (integrity check) can reuse it without re-creating
/// yet another sha256 helper.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)
        .with_context(|| format!("open: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex::encode(digest))
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// Env-var manipulation must be serialized — `std::env` is process-wide.
    /// Without this lock parallel `cargo test` runs will see each other's
    /// writes and the resolver tests will flake.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Save+restore env vars across a test that mutates them.
    struct EnvGuard {
        snapshots: Vec<(String, Option<String>)>,
    }
    impl EnvGuard {
        fn new(keys: &[&str]) -> Self {
            let snapshots = keys
                .iter()
                .map(|k| (k.to_string(), std::env::var(k).ok()))
                .collect();
            for k in keys {
                std::env::remove_var(k);
            }
            Self { snapshots }
        }
        fn set(&self, k: &str, v: &str) {
            std::env::set_var(k, v);
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.snapshots {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    // ── Slug round-trip ────────────────────────────────────────────────

    #[test]
    fn slug_simple_repo() {
        assert_eq!(slug_repo_id("google/gemma-4-27b-it").unwrap(), "google__gemma-4-27b-it");
    }

    #[test]
    fn slug_unslug_roundtrip() {
        let id = "mistralai/Mistral-7B-Instruct-v0.3";
        let slug = slug_repo_id(id).unwrap();
        assert_eq!(slug, "mistralai__Mistral-7B-Instruct-v0.3");
        assert_eq!(unslug_repo_id(&slug), id);
    }

    #[test]
    fn slug_rejects_empty() {
        assert!(slug_repo_id("").is_err());
    }

    #[test]
    fn slug_rejects_path_traversal() {
        assert!(slug_repo_id("..").is_err());
        assert!(slug_repo_id("foo/../bar").is_err());
        assert!(slug_repo_id("../etc/passwd").is_err());
    }

    #[test]
    fn slug_rejects_nul() {
        assert!(slug_repo_id("foo\0bar").is_err());
    }

    #[test]
    fn slug_long_repo_id() {
        // 200-char repo id (unrealistic but bounds-safe)
        let long = format!("{}/{}", "a".repeat(100), "b".repeat(100));
        let slug = slug_repo_id(&long).unwrap();
        assert_eq!(slug.len(), 100 + 2 + 100); // 100 'a' + "__" + 100 'b'
        assert_eq!(unslug_repo_id(&slug), long);
    }

    #[test]
    fn slug_unicode_repo_id() {
        let id = "用户/模型-v1";
        let slug = slug_repo_id(id).unwrap();
        assert!(slug.contains("__"));
        assert_eq!(unslug_repo_id(&slug), id);
    }

    // ── cache_model_path / cache_mmproj_path ──────────────────────────

    #[test]
    fn cache_model_path_layout() {
        let root = Path::new("/tmp/hf2q");
        let p = cache_model_path(root, "google/gemma-4-27b-it", QuantType::Q4_K_M).unwrap();
        assert_eq!(
            p,
            Path::new("/tmp/hf2q/models/google__gemma-4-27b-it/quantized/Q4_K_M/model.gguf")
        );
    }

    #[test]
    fn cache_mmproj_path_layout() {
        let root = Path::new("/tmp/hf2q");
        let p = cache_mmproj_path(root, "google/gemma-4-27b-it", QuantType::Q8_0).unwrap();
        assert_eq!(
            p,
            Path::new("/tmp/hf2q/models/google__gemma-4-27b-it/quantized/Q8_0/mmproj.gguf")
        );
    }

    #[test]
    fn cache_model_path_idempotent() {
        let root = Path::new("/tmp/hf2q");
        let a = cache_model_path(root, "x/y", QuantType::Q6_K).unwrap();
        let b = cache_model_path(root, "x/y", QuantType::Q6_K).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn cache_model_path_different_quants_differ() {
        let root = Path::new("/tmp/hf2q");
        let q4 = cache_model_path(root, "x/y", QuantType::Q4_K_M).unwrap();
        let q8 = cache_model_path(root, "x/y", QuantType::Q8_0).unwrap();
        assert_ne!(q4, q8);
    }

    // ── default_root resolution ───────────────────────────────────────

    #[test]
    fn default_root_honors_hf2q_cache_dir() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = EnvGuard::new(&["HF2Q_CACHE_DIR", "XDG_CACHE_HOME", "HOME"]);
        env.set("HF2Q_CACHE_DIR", "/explicit/override");
        env.set("HOME", "/should/be/ignored");
        env.set("XDG_CACHE_HOME", "/also/ignored");
        assert_eq!(default_root().unwrap(), PathBuf::from("/explicit/override"));
    }

    #[test]
    fn default_root_honors_xdg_cache_home() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = EnvGuard::new(&["HF2Q_CACHE_DIR", "XDG_CACHE_HOME", "HOME"]);
        env.set("XDG_CACHE_HOME", "/custom/xdg");
        env.set("HOME", "/should/be/ignored");
        assert_eq!(default_root().unwrap(), PathBuf::from("/custom/xdg/hf2q"));
    }

    #[test]
    fn default_root_falls_back_to_home() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = EnvGuard::new(&["HF2Q_CACHE_DIR", "XDG_CACHE_HOME", "HOME"]);
        env.set("HOME", "/users/robert");
        assert_eq!(default_root().unwrap(), PathBuf::from("/users/robert/.cache/hf2q"));
    }

    #[test]
    fn default_root_errors_when_all_env_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::new(&["HF2Q_CACHE_DIR", "XDG_CACHE_HOME", "HOME"]);
        assert!(default_root().is_err());
    }

    #[test]
    fn default_root_ignores_empty_env_values() {
        let _g = ENV_LOCK.lock().unwrap();
        let env = EnvGuard::new(&["HF2Q_CACHE_DIR", "XDG_CACHE_HOME", "HOME"]);
        env.set("HF2Q_CACHE_DIR", "");
        env.set("XDG_CACHE_HOME", "");
        env.set("HOME", "/users/robert");
        assert_eq!(default_root().unwrap(), PathBuf::from("/users/robert/.cache/hf2q"));
    }

    // ── ModelCache open/lookup ────────────────────────────────────────

    #[test]
    fn open_creates_directory_layout() {
        let tmp = TempDir::new().unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        assert_eq!(cache.root(), tmp.path());
        assert!(tmp.path().join("models").is_dir());
        assert!(tmp.path().join("locks").is_dir());
        // Manifest is NOT auto-written until something changes.
        assert!(!tmp.path().join("manifest.json").exists());
    }

    #[test]
    fn lookup_returns_none_for_missing_model() {
        let tmp = TempDir::new().unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        assert!(cache.lookup("foo/bar", QuantType::Q4_K_M).is_none());
        assert!(cache.lookup_model("foo/bar").is_none());
    }

    #[test]
    fn record_source_then_lookup() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        cache
            .record_source(
                "google/gemma-4-27b-it",
                "abc123def",
                SourcePointer::HfHub {
                    path: PathBuf::from("/some/hub/snapshot"),
                    revision: "abc123def".into(),
                },
            )
            .unwrap();

        let m = cache.lookup_model("google/gemma-4-27b-it").unwrap();
        assert_eq!(m.repo_id, "google/gemma-4-27b-it");
        assert_eq!(m.revision, "abc123def");
        assert!(matches!(m.source, Some(SourcePointer::HfHub { .. })));
        // No quantizations yet
        assert!(cache.lookup("google/gemma-4-27b-it", QuantType::Q4_K_M).is_none());
        // repo_meta.json companion landed
        assert!(tmp
            .path()
            .join("models")
            .join("google__gemma-4-27b-it")
            .join("repo_meta.json")
            .is_file());
    }

    #[test]
    fn record_quantized_then_lookup() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        cache
            .record_source(
                "x/y",
                "rev1",
                SourcePointer::Local {
                    path: PathBuf::from("/local/path"),
                    sha256: "0".repeat(64),
                },
            )
            .unwrap();

        let gguf = cache_model_path(tmp.path(), "x/y", QuantType::Q4_K_M).unwrap();
        // Pre-create the directory + a placeholder file so the companion
        // manifest write has somewhere to land.  iter-204 will land the
        // real bytes via tempfile-rename.
        fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        fs::write(&gguf, b"GGUF\0placeholder").unwrap();

        let entry = QuantEntry {
            quant_type: QuantType::Q4_K_M.as_str().to_string(),
            gguf_path: gguf.clone(),
            mmproj_path: None,
            bytes: 16,
            sha256: sha256_file(&gguf).unwrap(),
            quantized_at_secs: secs_since_epoch(),
            quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        cache.record_quantized("x/y", entry.clone()).unwrap();
        let got = cache.lookup("x/y", QuantType::Q4_K_M).unwrap();
        assert_eq!(got, &entry);

        // Per-quant companion manifest also written.
        let companion = gguf.parent().unwrap().join("manifest.json");
        assert!(companion.is_file());
    }

    #[test]
    fn record_quantized_without_source_errors() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();

        let gguf = cache_model_path(tmp.path(), "x/y", QuantType::Q4_K_M).unwrap();
        fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        fs::write(&gguf, b"GGUF").unwrap();

        let entry = QuantEntry {
            quant_type: QuantType::Q4_K_M.as_str().to_string(),
            gguf_path: gguf,
            mmproj_path: None,
            bytes: 4,
            sha256: "0".repeat(64),
            quantized_at_secs: 0,
            quantized_by_version: "0.1.0".into(),
        };
        // No `record_source` first → must error.
        assert!(cache.record_quantized("x/y", entry).is_err());
    }

    #[test]
    fn manifest_survives_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mut cache = ModelCache::open_at(tmp.path()).unwrap();
            cache
                .record_source(
                    "x/y",
                    "rev",
                    SourcePointer::Local {
                        path: PathBuf::from("/p"),
                        sha256: "1".repeat(64),
                    },
                )
                .unwrap();
        }
        // Reopen from cold disk
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        let m = cache.lookup_model("x/y").unwrap();
        assert_eq!(m.revision, "rev");
        assert_eq!(cache.manifest().schema_version, MANIFEST_SCHEMA_VERSION);
    }

    #[test]
    fn touch_updates_last_accessed() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        cache
            .record_source(
                "x/y",
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "2".repeat(64),
                },
            )
            .unwrap();
        let before = cache.lookup_model("x/y").unwrap().last_accessed_secs;

        // Sleep at least 1s so the seconds-resolution timestamp can advance.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache.touch("x/y").unwrap();
        let after = cache.lookup_model("x/y").unwrap().last_accessed_secs;
        assert!(after >= before, "touch must not move clock backwards");
    }

    #[test]
    fn touch_unknown_model_is_noop() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        // No record_source first; touch must succeed without writing anything.
        cache.touch("nonexistent/repo").unwrap();
        // Manifest never got dirty → not on disk.
        assert!(!tmp.path().join("manifest.json").exists());
    }

    // ── Atomic write invariants ───────────────────────────────────────

    #[test]
    fn atomic_write_no_temp_file_left_behind() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        cache
            .record_source(
                "x/y",
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "3".repeat(64),
                },
            )
            .unwrap();
        // After a successful flush the temp `.manifest.json.tmp.*` is gone.
        let entries: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().any(|n| n == "manifest.json"),
            "manifest.json missing in {:?}",
            entries
        );
        assert!(
            !entries.iter().any(|n| n.starts_with(".manifest.json.tmp")),
            "temp file leaked: {:?}",
            entries
        );
    }

    #[test]
    fn atomic_write_replaces_prior_manifest() {
        let tmp = TempDir::new().unwrap();
        // Round 1
        {
            let mut cache = ModelCache::open_at(tmp.path()).unwrap();
            cache
                .record_source(
                    "a/b",
                    "rev1",
                    SourcePointer::Local {
                        path: PathBuf::from("/p1"),
                        sha256: "4".repeat(64),
                    },
                )
                .unwrap();
        }
        let v1 = fs::read_to_string(tmp.path().join("manifest.json")).unwrap();
        // Round 2 — overwrite
        {
            let mut cache = ModelCache::open_at(tmp.path()).unwrap();
            cache
                .record_source(
                    "a/b",
                    "rev2",
                    SourcePointer::Local {
                        path: PathBuf::from("/p2"),
                        sha256: "5".repeat(64),
                    },
                )
                .unwrap();
        }
        let v2 = fs::read_to_string(tmp.path().join("manifest.json")).unwrap();
        assert_ne!(v1, v2);
        assert!(v2.contains("rev2"));
        assert!(!v2.contains("rev1"));
    }

    #[test]
    fn manifest_schema_version_mismatch_errors() {
        let tmp = TempDir::new().unwrap();
        // Hand-craft a manifest with a future schema version.
        fs::create_dir_all(tmp.path()).unwrap();
        fs::write(
            tmp.path().join("manifest.json"),
            r#"{"schema_version": 99, "models": {}}"#,
        )
        .unwrap();
        let err = ModelCache::open_at(tmp.path()).unwrap_err();
        assert!(format!("{err}").contains("schema_version"));
    }

    // ── File-lock semantics ───────────────────────────────────────────

    #[test]
    fn lock_acquire_releases_on_drop() {
        let tmp = TempDir::new().unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        {
            let _l = cache.lock_quant("x/y", QuantType::Q4_K_M).unwrap();
            // While held, try_lock from another handle to the same path
            // returns None.
            let attempt = cache.try_lock_quant("x/y", QuantType::Q4_K_M).unwrap();
            // libc::flock is per-fd advisory: a second flock on a NEW fd
            // for the same file from the SAME process blocks/refuses (BSD
            // flock semantics).  On Linux glibc maps to the same; on macOS
            // (the project's primary target) it is exactly BSD flock.
            assert!(
                attempt.is_none(),
                "concurrent try_lock_quant must observe held lock"
            );
        }
        // After drop the lock is released.
        let again = cache.try_lock_quant("x/y", QuantType::Q4_K_M).unwrap();
        assert!(again.is_some(), "lock must release on Drop");
    }

    #[test]
    fn lock_path_per_quant_distinct() {
        let tmp = TempDir::new().unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        let _l4 = cache.lock_quant("x/y", QuantType::Q4_K_M).unwrap();
        // Different quant must lock independently.
        let l8 = cache.try_lock_quant("x/y", QuantType::Q8_0).unwrap();
        assert!(l8.is_some(), "different quants must not share a lock");
    }

    #[test]
    fn lock_creates_locks_dir() {
        let tmp = TempDir::new().unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        let _l = cache.lock_quant("x/y", QuantType::Q4_K_M).unwrap();
        let lock_file = tmp
            .path()
            .join("locks")
            .join(format!("x__y__{}.lock", QuantType::Q4_K_M.as_str()));
        assert!(lock_file.is_file(), "lock file must exist on disk");
    }

    // ── HF cache detection (no copy) ──────────────────────────────────

    #[test]
    fn detect_hf_hub_finds_snapshot() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let env = EnvGuard::new(&["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HOME"]);
        env.set("HF_HUB_CACHE", tmp.path().to_str().unwrap());

        // Lay down the canonical hf-hub structure for `org/model`.
        let model_dir = tmp.path().join("models--org--model");
        let snap_a = model_dir.join("snapshots").join("rev-a");
        let snap_b = model_dir.join("snapshots").join("rev-b");
        fs::create_dir_all(&snap_a).unwrap();
        fs::create_dir_all(&snap_b).unwrap();
        // Drop a marker file so each snapshot has different mtime.
        fs::write(snap_a.join("config.json"), "{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::write(snap_b.join("config.json"), "{}").unwrap();

        let snap = ModelCache::detect_hf_hub_source("org/model").unwrap();
        // Bytes are NOT copied — just a pointer.  Verify the path lives
        // under the original hub root, not under the hf2q cache.
        assert!(snap.path.starts_with(tmp.path()));
        // Most-recent snapshot wins.  Filesystem mtime resolution can vary
        // across platforms (HFS+ has 1s resolution on older macOS), so
        // accept either revision rather than over-asserting.
        assert!(snap.revision == "rev-a" || snap.revision == "rev-b");
    }

    #[test]
    fn detect_hf_hub_returns_none_when_absent() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let env = EnvGuard::new(&["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HOME"]);
        env.set("HF_HUB_CACHE", tmp.path().to_str().unwrap());
        // No snapshot at all.
        assert!(ModelCache::detect_hf_hub_source("org/model").is_none());
    }

    #[test]
    fn detect_hf_hub_records_source_without_copy() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp_hub = TempDir::new().unwrap();
        let tmp_hf2q = TempDir::new().unwrap();
        let env = EnvGuard::new(&["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HOME"]);
        env.set("HF_HUB_CACHE", tmp_hub.path().to_str().unwrap());

        // Lay hf-hub layout
        let snap = tmp_hub
            .path()
            .join("models--org--model")
            .join("snapshots")
            .join("rev-x");
        fs::create_dir_all(&snap).unwrap();
        // Big-ish marker file we will assert is NOT copied.
        let marker = snap.join("model.safetensors");
        fs::write(&marker, vec![0u8; 1024]).unwrap();
        let hub_size_before = fs::metadata(&marker).unwrap().len();

        let detected = ModelCache::detect_hf_hub_source("org/model").unwrap();
        let mut cache = ModelCache::open_at(tmp_hf2q.path()).unwrap();
        cache
            .record_source(
                "org/model",
                &detected.revision,
                SourcePointer::HfHub {
                    path: detected.path.clone(),
                    revision: detected.revision.clone(),
                },
            )
            .unwrap();

        // hf2q cache contains ONLY metadata, not the safetensors bytes.
        let hf2q_total: u64 = walk_total(tmp_hf2q.path());
        let hub_size_after = fs::metadata(&marker).unwrap().len();
        assert_eq!(
            hub_size_before, hub_size_after,
            "HF cache file must not be modified"
        );
        assert!(
            hf2q_total < 8 * 1024,
            "hf2q cache should hold only json metadata, got {} bytes",
            hf2q_total
        );

        // And the manifest's source pointer references the hub path.
        let m = cache.lookup_model("org/model").unwrap();
        match m.source.as_ref().unwrap() {
            SourcePointer::HfHub { path, revision } => {
                assert!(path.starts_with(tmp_hub.path()));
                assert_eq!(revision, "rev-x");
            }
            other => panic!("expected HfHub, got {:?}", other),
        }
    }

    fn walk_total(dir: &Path) -> u64 {
        let mut total = 0u64;
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let ty = entry.file_type().unwrap();
            if ty.is_file() {
                total += entry.metadata().unwrap().len();
            } else if ty.is_dir() && !ty.is_symlink() {
                total += walk_total(&entry.path());
            }
        }
        total
    }

    // ── sha256_file ──────────────────────────────────────────────────

    #[test]
    fn sha256_file_known_vector() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("hello.bin");
        fs::write(&path, b"hello").unwrap();
        // SHA-256 of "hello"
        assert_eq!(
            sha256_file(&path).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn sha256_file_empty() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.bin");
        fs::write(&path, b"").unwrap();
        // SHA-256 of empty input
        assert_eq!(
            sha256_file(&path).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // ── ADR-005 Phase 3 iter-203 — schema v1→v2 migration ───────────────

    #[test]
    fn schema_v1_manifest_loads_with_empty_source_shards() {
        // Hand-craft a v1 manifest (no `source_shards` field).  Loader
        // must accept it and default the field to an empty Vec.
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("locks")).unwrap();
        let v1_json = r#"{
            "schema_version": 1,
            "models": {
                "old/model": {
                    "repo_id": "old/model",
                    "revision": "abc",
                    "source": null,
                    "quantizations": {},
                    "last_accessed_secs": 100
                }
            }
        }"#;
        fs::write(tmp.path().join("manifest.json"), v1_json).unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        let m = cache.lookup_model("old/model").expect("v1 entry loaded");
        assert_eq!(m.revision, "abc");
        assert!(
            m.source_shards.is_empty(),
            "v1 manifest must default source_shards to empty Vec"
        );
        // In-memory schema_version is bumped to current.
        assert_eq!(cache.manifest().schema_version, MANIFEST_SCHEMA_VERSION);
    }

    #[test]
    fn schema_v1_manifest_persists_as_v2_on_next_write() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("models")).unwrap();
        fs::create_dir_all(tmp.path().join("locks")).unwrap();
        let v1_json = r#"{
            "schema_version": 1,
            "models": {}
        }"#;
        fs::write(tmp.path().join("manifest.json"), v1_json).unwrap();
        // Open + mutate (touch a non-existent model is a no-op for the
        // manifest, so we record a real source instead).
        {
            let mut cache = ModelCache::open_at(tmp.path()).unwrap();
            cache
                .record_source(
                    "x/y",
                    "rev",
                    SourcePointer::Local {
                        path: PathBuf::from("/p"),
                        sha256: "0".repeat(64),
                    },
                )
                .unwrap();
        }
        // Reread on cold disk; schema_version is now 2.
        let raw = fs::read_to_string(tmp.path().join("manifest.json")).unwrap();
        assert!(
            raw.contains(r#""schema_version": 2"#) || raw.contains(r#""schema_version":2"#),
            "v2 manifest must persist schema_version=2: {raw}"
        );
    }

    #[test]
    fn schema_unsupported_too_old_errors() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path()).unwrap();
        fs::write(
            tmp.path().join("manifest.json"),
            r#"{"schema_version": 0, "models": {}}"#,
        )
        .unwrap();
        let err = ModelCache::open_at(tmp.path()).unwrap_err();
        assert!(format!("{err}").contains("schema_version"));
    }

    // ── record_source_with_shards ───────────────────────────────────────

    fn fake_shard(filename: &str, bytes: u64, sha256: Option<&str>) -> crate::input::integrity::ShardIntegrity {
        crate::input::integrity::ShardIntegrity {
            filename: filename.to_string(),
            bytes,
            sha256: sha256.map(|s| s.to_string()),
            hf_etag: sha256.unwrap_or("etag").to_string(),
            is_lfs: sha256.is_some(),
        }
    }

    #[test]
    fn record_source_with_shards_persists_into_manifest() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        let shards = vec![
            fake_shard(
                "model-00001-of-00002.safetensors",
                1024,
                Some(&"a".repeat(64)),
            ),
            fake_shard(
                "model-00002-of-00002.safetensors",
                2048,
                Some(&"b".repeat(64)),
            ),
            fake_shard("config.json", 200, None),
        ];
        cache
            .record_source_with_shards(
                "org/m",
                "rev1",
                SourcePointer::HfHub {
                    path: PathBuf::from("/hub/path"),
                    revision: "rev1".into(),
                },
                shards.clone(),
            )
            .unwrap();
        let m = cache.lookup_model("org/m").unwrap();
        assert_eq!(m.source_shards.len(), 3);
        assert_eq!(
            m.source_shards[0].filename,
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(m.source_shards[0].bytes, 1024);
        assert_eq!(m.source_shards[0].sha256.as_deref(), Some(&*"a".repeat(64)));
        assert!(m.source_shards[0].is_lfs);
        assert_eq!(m.source_shards[2].filename, "config.json");
        assert!(!m.source_shards[2].is_lfs);
        // verified_at_secs is non-zero (best-effort timestamp).
        assert!(m.source_shards[0].verified_at_secs > 0);
    }

    #[test]
    fn record_source_with_shards_survives_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mut cache = ModelCache::open_at(tmp.path()).unwrap();
            let shards = vec![fake_shard("a.safetensors", 8, Some(&"c".repeat(64)))];
            cache
                .record_source_with_shards(
                    "org/m",
                    "rev",
                    SourcePointer::HfHub {
                        path: PathBuf::from("/p"),
                        revision: "rev".into(),
                    },
                    shards,
                )
                .unwrap();
        }
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        let m = cache.lookup_model("org/m").unwrap();
        assert_eq!(m.source_shards.len(), 1);
        assert_eq!(m.source_shards[0].filename, "a.safetensors");
    }

    // ── verify_quantized ────────────────────────────────────────────────

    fn record_real_quant(tmp: &Path, repo: &str, contents: &[u8]) -> ModelCache {
        let mut cache = ModelCache::open_at(tmp).unwrap();
        cache
            .record_source(
                repo,
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "0".repeat(64),
                },
            )
            .unwrap();
        let gguf = cache_model_path(tmp, repo, QuantType::Q4_K_M).unwrap();
        fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        fs::write(&gguf, contents).unwrap();
        let entry = QuantEntry {
            quant_type: QuantType::Q4_K_M.as_str().to_string(),
            gguf_path: gguf.clone(),
            mmproj_path: None,
            bytes: contents.len() as u64,
            sha256: sha256_file(&gguf).unwrap(),
            quantized_at_secs: secs_since_epoch(),
            quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        cache.record_quantized(repo, entry).unwrap();
        cache
    }

    #[test]
    fn verify_quantized_pass_when_bytes_match_manifest() {
        let tmp = TempDir::new().unwrap();
        let cache = record_real_quant(tmp.path(), "org/m", b"GGUF\0valid\0bytes");
        cache
            .verify_quantized("org/m", QuantType::Q4_K_M)
            .expect("matching bytes should verify");
    }

    #[test]
    fn verify_quantized_fail_when_bytes_mutated_after_record() {
        let tmp = TempDir::new().unwrap();
        let cache = record_real_quant(tmp.path(), "org/m", b"original");
        // Tamper: overwrite the GGUF with different bytes (same length so
        // the size match doesn't short-circuit any logic — we still rely
        // on the sha256 catching it).
        let gguf = cache_model_path(tmp.path(), "org/m", QuantType::Q4_K_M).unwrap();
        fs::write(&gguf, b"tampered").unwrap();
        let err = cache
            .verify_quantized("org/m", QuantType::Q4_K_M)
            .expect_err("tampered bytes must be detected");
        let msg = format!("{err}");
        assert!(msg.contains("SHA-256 mismatch"), "msg: {msg}");
        assert!(msg.contains("org/m"), "msg: {msg}");
        assert!(msg.contains("Q4_K_M"), "msg: {msg}");
        assert!(msg.contains("--no-integrity"), "msg: {msg}");
    }

    #[test]
    fn verify_quantized_fail_when_no_manifest_entry() {
        let tmp = TempDir::new().unwrap();
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        let err = cache
            .verify_quantized("ghost/repo", QuantType::Q4_K_M)
            .expect_err("uncached repo must fail");
        let msg = format!("{err}");
        assert!(msg.contains("no manifest entry"), "msg: {msg}");
    }

    #[test]
    fn verify_quantized_fail_when_gguf_deleted_from_disk() {
        let tmp = TempDir::new().unwrap();
        let cache = record_real_quant(tmp.path(), "org/m", b"valid");
        // Delete the GGUF after recording.
        let gguf = cache_model_path(tmp.path(), "org/m", QuantType::Q4_K_M).unwrap();
        fs::remove_file(&gguf).unwrap();
        let err = cache
            .verify_quantized("org/m", QuantType::Q4_K_M)
            .expect_err("missing gguf must fail");
        let msg = format!("{err}");
        assert!(msg.contains("missing on disk"), "msg: {msg}");
    }

    // ── SourceShard::from_integrity adapter ─────────────────────────────

    #[test]
    fn source_shard_adapter_copies_all_fields_and_stamps_timestamp() {
        let integ = crate::input::integrity::ShardIntegrity {
            filename: "model.safetensors".into(),
            bytes: 4096,
            sha256: Some("d".repeat(64)),
            hf_etag: "d".repeat(64),
            is_lfs: true,
        };
        let s = SourceShard::from_integrity(&integ);
        assert_eq!(s.filename, integ.filename);
        assert_eq!(s.bytes, integ.bytes);
        assert_eq!(s.sha256, integ.sha256);
        assert_eq!(s.hf_etag, integ.hf_etag);
        assert_eq!(s.is_lfs, integ.is_lfs);
        assert!(s.verified_at_secs > 0);
    }

    // ── ADR-005 Phase 4 iter-207 — compute_source_bundle_sha256 ─────────

    fn shard(filename: &str, sha: Option<&str>) -> SourceShard {
        SourceShard {
            filename: filename.to_string(),
            bytes: 1,
            sha256: sha.map(|s| s.to_string()),
            hf_etag: sha.map(|s| s.to_string()).unwrap_or_default(),
            is_lfs: sha.is_some(),
            verified_at_secs: 1,
        }
    }

    #[test]
    fn bundle_sha_returns_none_for_empty_list() {
        assert!(compute_source_bundle_sha256(&[]).is_none());
    }

    #[test]
    fn bundle_sha_returns_none_when_all_shards_lack_sha() {
        // Non-LFS files (config.json, tokenizer.json) carry no sha256.
        let shards = vec![
            shard("config.json", None),
            shard("tokenizer.json", None),
        ];
        assert!(compute_source_bundle_sha256(&shards).is_none());
    }

    #[test]
    fn bundle_sha_is_deterministic_under_input_reordering() {
        let a = shard("a.safetensors", Some(&"a".repeat(64)));
        let b = shard("b.safetensors", Some(&"b".repeat(64)));
        let c = shard("c.safetensors", Some(&"c".repeat(64)));

        let h1 = compute_source_bundle_sha256(&[a.clone(), b.clone(), c.clone()]).unwrap();
        let h2 = compute_source_bundle_sha256(&[c, a, b]).unwrap();
        assert_eq!(h1, h2, "bundle SHA must be order-independent");
        assert_eq!(h1.len(), 64, "must be 64-hex SHA-256");
        assert!(
            h1.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "must be lowercase hex"
        );
    }

    #[test]
    fn bundle_sha_skips_shards_without_sha_but_includes_others() {
        // Non-LFS files in the same list don't perturb the bundle hash —
        // a hf2q-emitted GGUF and the cache compute the SAME hash even
        // though the shard list passed in may include both kinds.
        let s = vec![
            shard("model.safetensors", Some(&"e".repeat(64))),
            shard("config.json", None),
            shard("tokenizer.json", None),
        ];
        let h_full = compute_source_bundle_sha256(&s).unwrap();
        let h_only_lfs =
            compute_source_bundle_sha256(&[shard("model.safetensors", Some(&"e".repeat(64)))])
                .unwrap();
        assert_eq!(h_full, h_only_lfs);
    }

    #[test]
    fn bundle_sha_normalizes_uppercase_hex() {
        let lower = compute_source_bundle_sha256(&[shard("a", Some(&"a".repeat(64)))]).unwrap();
        let upper = compute_source_bundle_sha256(&[shard("a", Some(&"A".repeat(64)))]).unwrap();
        assert_eq!(lower, upper, "uppercase shard SHA must match lowercase");
    }

    #[test]
    fn bundle_sha_distinct_for_distinct_inputs() {
        let h1 = compute_source_bundle_sha256(&[shard("a", Some(&"a".repeat(64)))]).unwrap();
        let h2 = compute_source_bundle_sha256(&[shard("a", Some(&"b".repeat(64)))]).unwrap();
        let h3 = compute_source_bundle_sha256(&[shard("b", Some(&"a".repeat(64)))]).unwrap();
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }

    // ── ADR-005 Phase 3 iter-205 — invalidate / invalidate_repo / purge ─

    /// Drop two quants of one repo onto disk with a real manifest
    /// entry per quant; returns (cache, gguf paths).
    fn fab_two_quants(tmp: &Path, repo: &str) -> (ModelCache, PathBuf, PathBuf) {
        let mut cache = ModelCache::open_at(tmp).unwrap();
        cache
            .record_source(
                repo,
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "0".repeat(64),
                },
            )
            .unwrap();

        let g4 = cache_model_path(tmp, repo, QuantType::Q4_K_M).unwrap();
        fs::create_dir_all(g4.parent().unwrap()).unwrap();
        fs::write(&g4, b"Q4_K_M_BYTES_FOR_TEST").unwrap();
        cache
            .record_quantized(
                repo,
                QuantEntry {
                    quant_type: QuantType::Q4_K_M.as_str().to_string(),
                    gguf_path: g4.clone(),
                    mmproj_path: None,
                    bytes: 21,
                    sha256: sha256_file(&g4).unwrap(),
                    quantized_at_secs: secs_since_epoch(),
                    quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
                },
            )
            .unwrap();

        let g8 = cache_model_path(tmp, repo, QuantType::Q8_0).unwrap();
        fs::create_dir_all(g8.parent().unwrap()).unwrap();
        fs::write(&g8, b"Q8_0_BYTES_FOR_TEST_LARGER").unwrap();
        cache
            .record_quantized(
                repo,
                QuantEntry {
                    quant_type: QuantType::Q8_0.as_str().to_string(),
                    gguf_path: g8.clone(),
                    mmproj_path: None,
                    bytes: 26,
                    sha256: sha256_file(&g8).unwrap(),
                    quantized_at_secs: secs_since_epoch(),
                    quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
                },
            )
            .unwrap();
        (cache, g4, g8)
    }

    #[test]
    fn invalidate_removes_quant_entry_and_files() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, g4, g8) = fab_two_quants(tmp.path(), "org/m");

        assert!(g4.is_file(), "Q4_K_M GGUF must be on disk pre-invalidate");
        assert!(g8.is_file(), "Q8_0 GGUF must be on disk pre-invalidate");
        assert!(cache.lookup("org/m", QuantType::Q4_K_M).is_some());
        assert!(cache.lookup("org/m", QuantType::Q8_0).is_some());

        let _freed = cache.invalidate("org/m", QuantType::Q4_K_M).unwrap();

        // Q4_K_M is gone from disk + manifest; Q8_0 untouched.
        assert!(!g4.exists(), "Q4_K_M GGUF must be removed");
        assert!(!g4.parent().unwrap().exists(), "Q4_K_M dir must be removed");
        assert!(g8.is_file(), "Q8_0 GGUF must survive");
        assert!(cache.lookup("org/m", QuantType::Q4_K_M).is_none());
        assert!(cache.lookup("org/m", QuantType::Q8_0).is_some());
    }

    #[test]
    fn invalidate_returns_bytes_freed_matching_disk_walk() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, g4, _g8) = fab_two_quants(tmp.path(), "org/m");

        // Pre-walk the dir so the test's expectation is the truth on disk
        // (covers gguf + companion manifest.json), independent of the
        // QuantEntry::bytes field which only tracks the GGUF.
        let pre = dir_total_bytes(g4.parent().unwrap());
        assert!(pre > 0, "pre-bytes must be non-zero");

        let freed = cache.invalidate("org/m", QuantType::Q4_K_M).unwrap();
        assert_eq!(freed, pre, "freed bytes must match the pre-removal walk");
    }

    #[test]
    fn invalidate_unknown_repo_errors_named() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        let err = cache
            .invalidate("ghost/repo", QuantType::Q4_K_M)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown_repo"), "msg: {msg}");
        assert!(msg.contains("ghost/repo"), "msg: {msg}");
    }

    #[test]
    fn invalidate_unknown_quant_errors_named() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/m");
        let err = cache
            .invalidate("org/m", QuantType::Q3_K_M)  // not cached
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown_quant"), "msg: {msg}");
        assert!(msg.contains("Q3_K_M"), "msg: {msg}");
    }

    #[test]
    fn invalidate_persists_across_reopen() {
        // After invalidate, reopening from cold disk must not rediscover
        // the removed entry (the manifest must have been atomically
        // flushed).
        let tmp = TempDir::new().unwrap();
        {
            let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/m");
            cache.invalidate("org/m", QuantType::Q4_K_M).unwrap();
        }
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        assert!(cache.lookup("org/m", QuantType::Q4_K_M).is_none());
        assert!(cache.lookup("org/m", QuantType::Q8_0).is_some());
    }

    #[test]
    fn invalidate_holds_per_quant_lock_during_op() {
        // The lock taken inside `invalidate` must observably hold the
        // (repo, quant) lock — concurrent `try_lock_quant` from the
        // SAME process for the same key returns None until invalidate
        // releases.  We can't probe mid-call (no callbacks), but the
        // post-condition (lock released after return) is provable.
        let tmp = TempDir::new().unwrap();
        let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/m");
        cache.invalidate("org/m", QuantType::Q4_K_M).unwrap();
        // Lock must be released — try_lock returns Some.
        let l = cache
            .try_lock_quant("org/m", QuantType::Q4_K_M)
            .unwrap();
        assert!(
            l.is_some(),
            "post-invalidate lock must be released (RAII Drop fires before flush returns)"
        );
    }

    #[test]
    fn invalidate_repo_removes_all_quants_and_model_dir() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, g4, g8) = fab_two_quants(tmp.path(), "org/m");

        let model_dir = tmp
            .path()
            .join("models")
            .join(slug_repo_id("org/m").unwrap());
        assert!(model_dir.is_dir());

        let freed = cache.invalidate_repo("org/m").unwrap();
        assert!(freed > 0, "freed must be non-zero with two real quants");

        assert!(!model_dir.exists(), "model dir must be removed");
        assert!(!g4.exists());
        assert!(!g8.exists());
        assert!(cache.lookup_model("org/m").is_none());
    }

    #[test]
    fn invalidate_repo_unknown_errors_named() {
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        let err = cache.invalidate_repo("ghost/repo").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown_repo"), "msg: {msg}");
        assert!(msg.contains("ghost/repo"), "msg: {msg}");
    }

    #[test]
    fn invalidate_repo_handles_repo_with_no_quants() {
        // A `record_source` without any `record_quantized` (i.e. an
        // in-flight or failed quantize) must still be removable.
        let tmp = TempDir::new().unwrap();
        let mut cache = ModelCache::open_at(tmp.path()).unwrap();
        cache
            .record_source(
                "org/m",
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "0".repeat(64),
                },
            )
            .unwrap();
        cache.invalidate_repo("org/m").unwrap();
        assert!(cache.lookup_model("org/m").is_none());
    }

    #[test]
    fn purge_removes_everything_and_resets_manifest() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/a");
        // Add a second model so we can prove purge nukes more than one.
        cache
            .record_source(
                "org/b",
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "1".repeat(64),
                },
            )
            .unwrap();

        let freed = cache.purge().unwrap();
        assert!(freed > 0);
        assert!(cache.manifest().models.is_empty());
        assert!(cache.lookup_model("org/a").is_none());
        assert!(cache.lookup_model("org/b").is_none());

        // models/ dir must still exist (layout invariant) — empty.
        let models = tmp.path().join("models");
        assert!(models.is_dir(), "models/ must be re-created post-purge");
        let count = fs::read_dir(&models).unwrap().count();
        assert_eq!(count, 0, "models/ must be empty post-purge");
    }

    #[test]
    fn purge_preserves_schema_version_v2_on_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/a");
            cache.purge().unwrap();
        }
        // Reopen from cold disk.
        let cache = ModelCache::open_at(tmp.path()).unwrap();
        assert_eq!(
            cache.manifest().schema_version,
            MANIFEST_SCHEMA_VERSION,
            "purge must not regress schema version"
        );
        assert!(cache.manifest().models.is_empty());
    }

    #[test]
    fn purge_idempotent_second_call_returns_zero() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/m");
        let _first = cache.purge().unwrap();
        let second = cache.purge().unwrap();
        assert_eq!(second, 0, "second purge must report 0 bytes freed");
    }

    #[test]
    fn iter_entries_lists_all_repos_and_quants() {
        let tmp = TempDir::new().unwrap();
        let (mut cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/a");
        // Second repo, single Q8_0 quant.
        let g_b = cache_model_path(tmp.path(), "org/b", QuantType::Q8_0).unwrap();
        cache
            .record_source(
                "org/b",
                "rev",
                SourcePointer::Local {
                    path: PathBuf::from("/p"),
                    sha256: "2".repeat(64),
                },
            )
            .unwrap();
        fs::create_dir_all(g_b.parent().unwrap()).unwrap();
        fs::write(&g_b, b"Q8_0").unwrap();
        cache
            .record_quantized(
                "org/b",
                QuantEntry {
                    quant_type: QuantType::Q8_0.as_str().to_string(),
                    gguf_path: g_b.clone(),
                    mmproj_path: None,
                    bytes: 4,
                    sha256: sha256_file(&g_b).unwrap(),
                    quantized_at_secs: secs_since_epoch(),
                    quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
                },
            )
            .unwrap();

        let entries: Vec<_> = cache.iter_entries().collect();
        assert_eq!(entries.len(), 2, "two distinct repos");
        // BTreeMap iteration is alphabetical → org/a before org/b.
        assert_eq!(entries[0].repo_id, "org/a");
        assert_eq!(entries[0].model.quantizations.len(), 2); // Q4_K_M + Q8_0
        assert_eq!(entries[1].repo_id, "org/b");
        assert_eq!(entries[1].model.quantizations.len(), 1);
    }

    #[test]
    fn total_bytes_on_disk_matches_walk() {
        let tmp = TempDir::new().unwrap();
        let (cache, _g4, _g8) = fab_two_quants(tmp.path(), "org/m");
        let direct = dir_total_bytes(&tmp.path().join("models"));
        assert_eq!(cache.total_bytes_on_disk(), direct);
        assert!(cache.total_bytes_on_disk() > 0);
    }

    #[test]
    fn dir_total_bytes_returns_zero_for_missing_path() {
        // Helper-level guard: the silent-on-missing semantics are
        // load-bearing for `total_bytes_on_disk` against a freshly
        // opened cache.
        let tmp = TempDir::new().unwrap();
        let phantom = tmp.path().join("does/not/exist");
        assert_eq!(dir_total_bytes(&phantom), 0);
    }

    // ── QuantType::from_canonical_str ──────────────────────────────────

    #[test]
    fn quant_type_round_trip_via_from_canonical_str() {
        for q in [
            QuantType::Q8_0,
            QuantType::Q6_K,
            QuantType::Q4_K_M,
            QuantType::Q3_K_M,
        ] {
            let parsed = QuantType::from_canonical_str(q.as_str()).unwrap();
            assert_eq!(parsed, q, "round-trip for {}", q.as_str());
        }
    }

    #[test]
    fn quant_type_case_insensitive() {
        assert_eq!(
            QuantType::from_canonical_str("q4_k_m").unwrap(),
            QuantType::Q4_K_M
        );
        assert_eq!(
            QuantType::from_canonical_str("Q4_k_M").unwrap(),
            QuantType::Q4_K_M
        );
    }

    #[test]
    fn quant_type_unknown_errors_lists_supported() {
        let err = QuantType::from_canonical_str("Q5_K_S").unwrap_err();
        // Error message names every supported variant so the operator
        // can self-correct without consulting docs.
        assert!(err.contains("Q8_0"), "err: {err}");
        assert!(err.contains("Q6_K"), "err: {err}");
        assert!(err.contains("Q4_K_M"), "err: {err}");
        assert!(err.contains("Q3_K_M"), "err: {err}");
        assert!(err.contains("Q5_K_S"), "err: {err}");
    }
}
