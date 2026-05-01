//! ADR-017 §R-F6 — operator-facing KV-cache namespace ops.
//!
//! This module backs `hf2q cache --kv-namespace {list,size,clear}`,
//! closing `docs/operating-kv-cache.md` §11 #4. It deliberately
//! scopes to filesystem walks over the on-disk layout documented in
//! §3 of the runbook (and in `block_store.rs::block_path` /
//! `quarantine_dir`):
//!
//! ```text
//! <PATH>/
//!   locks/                              # advisory flock files (NEVER touched by clear)
//!   models/<fp_short>/
//!     kv/<hex0>/<full_hex>.safetensors  # block envelopes
//!     kv-quarantine/<reason>__<orig>    # corrupted-block holding
//! ```
//!
//! ## Why a fresh module rather than reusing `cache::ModelCache`
//!
//! The weights-side `ModelCache` (`src/serve/cache.rs`) is
//! manifest-driven (it owns `cache_index.json`); the KV cache has
//! NO manifest — `BlockIndex` is rebuilt from disk by
//! `recover_from_disk`. The two trees share `models/` semantics but
//! not state-shape, so a thin fs-walker is the right primitive. The
//! walk pattern itself mirrors `index.rs::recover_from_disk` (lines
//! 177-180) at the directory layer; we just don't validate
//! envelopes — `cache --kv-namespace list` is a sizing report, not
//! a parity check.
//!
//! ## What this module does NOT do
//!
//! - Does not load any model. Pure fs-walk + `fs::remove_dir_all`.
//! - Does not invoke `compute_model_fingerprint`. The CLI's
//!   `--model <repo> [--quant <q>]` arguments are translated to
//!   `<fp_short>` by the same `compute_model_fingerprint(repo,
//!   quant, "", "", "")` rule the spiller uses
//!   (`spiller.rs:242-244`); see [`fp_short_for`].
//! - Does not write to `<root>/locks/`. The §11 #4 spec is explicit
//!   that lock files survive a cache clear (they're session-scoped,
//!   bound to the live `flock(LOCK_EX)` fds in
//!   `block_store.rs::AdvisoryLock`).
//! - Does not honour `<root>/models/<fp_short>/kv-quarantine/` as a
//!   separate clearable scope at this commit — `clear --model` removes
//!   the whole `<fp_short>/` directory (kv + kv-quarantine) since
//!   the operator's intent is "purge everything for this repo".

use std::fs;
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::serve::kv_persist::format::compute_model_fingerprint;
use crate::serve::quant_select::QuantType;

/// Env-var consulted as fallback when `--kv-path` is absent. See ADR-017
/// §R-F1 — the runbook documents the operator-facing convention.
pub const HF2Q_KV_PERSIST_PATH_ENV: &str = "HF2Q_KV_PERSIST_PATH";

/// Sentinel file at `<kv_root>/.cache_lock` — touched + flocked
/// non-blocking by the clear path so concurrent clears (and a
/// future serve-side root-lock) collide cleanly. See [`active_serve_guard`].
pub const ACTIVE_SERVE_SENTINEL: &str = ".cache_lock";

/// Resolve the kv-persist root: explicit path wins, else env var, else
/// error. No silent default — see CLI docstring on `CacheAction`.
pub fn resolve_kv_root(kv_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = kv_path {
        return Ok(p.to_path_buf());
    }
    match std::env::var(HF2Q_KV_PERSIST_PATH_ENV) {
        Ok(s) if !s.is_empty() => Ok(PathBuf::from(s)),
        _ => Err(anyhow::anyhow!(
            "hf2q cache --kv-namespace: kv-persist path not provided. \
             Pass --kv-path PATH or set {}=PATH (the path supplied to \
             `hf2q serve --kv-persist=PATH`).",
            HF2Q_KV_PERSIST_PATH_ENV
        )),
    }
}

/// Translate `(repo, quant)` into the same 16-hex `<fp_short>`
/// directory name the spiller writes
/// (`spiller.rs::family_model_fp` calls
/// `compute_model_fingerprint(repo, quant, "", "", "")`).
pub fn fp_short_for(repo: &str, quant: QuantType) -> String {
    compute_model_fingerprint(repo, quant.as_str(), "", "", "").short_hex()
}

/// One row of `cache --kv-namespace list` output. Per-`<fp_short>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvNamespaceEntry {
    /// 16-hex directory name under `<root>/models/`.
    pub fp_short: String,
    /// Total bytes-on-disk under `<root>/models/<fp_short>/` (kv + kv-quarantine).
    pub bytes_on_disk: u64,
    /// Count of `*.safetensors` files under `<root>/models/<fp_short>/kv/`.
    /// Excludes quarantine and `*.tmp.<pid>` scratch (recovery.rs:167-171).
    pub block_count: u64,
}

/// Walk `<root>/models/` and tally one [`KvNamespaceEntry`] per
/// child directory. Tolerates a missing root or a missing `models/`
/// subdir (returns `Ok(vec![])` — the operator may have run before
/// `cmd_serve --kv-persist=PATH` ever started).
pub fn list_namespaces(kv_root: &Path) -> Result<Vec<KvNamespaceEntry>> {
    let models_dir = kv_root.join("models");
    if !models_dir.exists() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for entry in fs::read_dir(&models_dir)
        .with_context(|| format!("read_dir {}", models_dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry under {}", models_dir.display()))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(fp_short) = path.file_name().and_then(|s| s.to_str()).map(str::to_owned) else {
            continue;
        };
        let bytes_on_disk = dir_total_bytes(&path);
        let block_count = count_block_files(&path.join("kv"));
        out.push(KvNamespaceEntry {
            fp_short,
            bytes_on_disk,
            block_count,
        });
    }
    out.sort_by(|a, b| a.fp_short.cmp(&b.fp_short));
    Ok(out)
}

/// Total bytes-on-disk under `<root>/models/`. Mirror of `list_namespaces`
/// summed; kept as a separate primitive so a `cache size --kv-namespace`
/// invocation skips per-entry allocation for very large caches.
pub fn total_bytes(kv_root: &Path) -> u64 {
    dir_total_bytes(&kv_root.join("models"))
}

/// Reason a `--kv-namespace clear` was refused.
#[derive(Debug)]
pub enum ClearRefusal {
    /// `--all` (or `--all --yes`) is not supported under `--kv-namespace`.
    AllNotSupported,
    /// `--model` was missing; per §11 #4 we refuse rather than silently
    /// deleting the whole tree.
    ModelMissing,
    /// Active-serve sentinel flock could not be acquired non-blocking.
    /// Pass `--force` to override.
    Locked,
}

/// Outcome of a successful `--kv-namespace clear`.
#[derive(Debug)]
pub struct ClearOutcome {
    /// 16-hex `<fp_short>` of the removed directory (target of the rm).
    pub fp_short: String,
    /// Bytes freed (computed pre-rm by walking the subtree).
    pub bytes_freed: u64,
    /// `true` when the target directory existed; `false` is idempotent
    /// no-op (operator passed `--model` for a repo that was never cached).
    pub existed: bool,
}

/// Implementation of `hf2q cache clear --kv-namespace --model <repo>
/// [--quant <q>]`. Removes ONE `<fp_short>` directory exactly, never
/// `locks/`, never sibling `<fp_short_other>` dirs.
///
/// `force = true` skips the active-serve sentinel-flock check. The
/// caller is responsible for ensuring `cmd_serve` is stopped.
pub fn clear_namespace(
    kv_root: &Path,
    repo: &str,
    quant: QuantType,
    force: bool,
) -> std::result::Result<ClearOutcome, ClearRefusalErr> {
    if !force {
        match active_serve_guard(kv_root) {
            Ok(_guard) => { /* drop releases */ }
            Err(e) => {
                return Err(ClearRefusalErr::Locked(format!(
                    "kv_root sentinel locked at {}: {}; another `hf2q` \
                     process appears to be holding it. Stop the serve \
                     first or pass --force.",
                    kv_root.join(ACTIVE_SERVE_SENTINEL).display(),
                    e
                )));
            }
        }
    }

    let fp_short = fp_short_for(repo, quant);
    let target = kv_root.join("models").join(&fp_short);
    let existed = target.exists();
    let bytes_freed = if existed { dir_total_bytes(&target) } else { 0 };

    if existed {
        fs::remove_dir_all(&target).map_err(|e| {
            ClearRefusalErr::Io(format!(
                "remove_dir_all({}): {}",
                target.display(),
                e
            ))
        })?;
    }

    Ok(ClearOutcome {
        fp_short,
        bytes_freed,
        existed,
    })
}

/// `clear_namespace` removing every quant variant cached for `repo`.
/// Iterates over all known `QuantType`s — kv-namespace fingerprints
/// only vary on `(repo, quant)` per §3 of the runbook.
pub fn clear_namespace_all_quants(
    kv_root: &Path,
    repo: &str,
    force: bool,
) -> std::result::Result<Vec<ClearOutcome>, ClearRefusalErr> {
    let _guard;
    if !force {
        _guard = active_serve_guard(kv_root).map_err(|e| {
            ClearRefusalErr::Locked(format!(
                "kv_root sentinel locked at {}: {}; another `hf2q` \
                 process appears to be holding it. Stop the serve \
                 first or pass --force.",
                kv_root.join(ACTIVE_SERVE_SENTINEL).display(),
                e
            ))
        })?;
    }

    let mut outcomes = Vec::new();
    for q in [
        QuantType::Q8_0,
        QuantType::Q6_K,
        QuantType::Q4_K_M,
        QuantType::Q3_K_M,
    ] {
        let fp_short = fp_short_for(repo, q);
        let target = kv_root.join("models").join(&fp_short);
        let existed = target.exists();
        let bytes_freed = if existed { dir_total_bytes(&target) } else { 0 };
        if existed {
            fs::remove_dir_all(&target).map_err(|e| {
                ClearRefusalErr::Io(format!(
                    "remove_dir_all({}): {}",
                    target.display(),
                    e
                ))
            })?;
        }
        outcomes.push(ClearOutcome {
            fp_short,
            bytes_freed,
            existed,
        });
    }
    Ok(outcomes)
}

/// Error variants returned from `clear_namespace*`. `anyhow::Error`
/// would lose the structured `Locked` discrimination tests check on,
/// so we expose a tiny enum.
#[derive(Debug)]
pub enum ClearRefusalErr {
    Locked(String),
    Io(String),
}

impl std::fmt::Display for ClearRefusalErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Locked(msg) | Self::Io(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for ClearRefusalErr {}

// ---------------------------------------------------------------------------
// Active-serve sentinel — non-blocking flock on `<kv_root>/.cache_lock`.
//
// At this commit `cmd_serve --kv-persist=PATH` does NOT lock the root
// (only per-block files via `block_store.rs::AdvisoryLock`). So this
// guard primarily protects against concurrent clear ops invoked from
// two operator shells; once a future iter adds root-side flock to
// `cmd_serve` (the natural extension), the same sentinel will refuse
// a clear under live serve.
// ---------------------------------------------------------------------------

/// RAII flock-on-drop. Returned to the caller so the lock outlives
/// the directory walk + remove.
pub struct ActiveServeGuard {
    _file: fs::File,
}

/// Acquire `<kv_root>/.cache_lock` via `flock(LOCK_EX | LOCK_NB)`.
/// Errors when the lock is already held by another process. Creates
/// the sentinel file (and `kv_root` if missing) lazily.
fn active_serve_guard(kv_root: &Path) -> io::Result<ActiveServeGuard> {
    if !kv_root.exists() {
        fs::create_dir_all(kv_root)?;
    }
    let path = kv_root.join(ACTIVE_SERVE_SENTINEL);
    let file = fs::File::options()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;
    let fd = file.as_raw_fd();
    // SAFETY: fd owned by `file`; flock is documented thread-safe.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(ActiveServeGuard { _file: file })
}

// ---------------------------------------------------------------------------
// Filesystem walk helpers — kept private to the module.
// ---------------------------------------------------------------------------

/// Recursive bytes-on-disk under `dir`. Returns `0` for missing /
/// permission-denied entries so the operator gets a value rather
/// than a hard error on a partially-populated cache. Mirrors
/// `serve::cache::dir_total_bytes` semantics.
fn dir_total_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = fs::read_dir(dir) else {
        return 0;
    };
    for entry in rd.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total = total.saturating_add(dir_total_bytes(&entry.path()));
        } else {
            total = total.saturating_add(meta.len());
        }
    }
    total
}

/// Count `*.safetensors` files under `kv_dir` (one level deep below the
/// hex-fanout buckets `<kv_dir>/<hex0>/`). Skips `*.tmp.<pid>` scratch
/// per `recovery.rs:167-171`.
fn count_block_files(kv_dir: &Path) -> u64 {
    let mut count = 0u64;
    let Ok(rd) = fs::read_dir(kv_dir) else {
        return 0;
    };
    for fanout in rd.flatten() {
        let path = fanout.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(inner) = fs::read_dir(&path) else { continue };
        for f in inner.flatten() {
            let p = f.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Tests — fs-mocked via tempdir; no real model loads, no GPU.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::SystemTime;

    fn temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "hf2q-cache-ops-{}-{}-{}-{}",
            label, pid, n, nanos
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn touch_block(kv_root: &Path, fp_short: &str, hex_full: &str, body: &[u8]) {
        let dir = kv_root
            .join("models")
            .join(fp_short)
            .join("kv")
            .join(&hex_full[..1]);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.safetensors", hex_full));
        fs::write(&path, body).unwrap();
    }

    fn touch_lockdir(kv_root: &Path, fp_short: &str) {
        // Mirror `block_store.rs::new_with_index` lines 117-121.
        fs::create_dir_all(kv_root.join("locks")).unwrap();
        fs::write(
            kv_root
                .join("locks")
                .join(format!("{}__ab.lock", fp_short)),
            b"",
        )
        .unwrap();
    }

    // ── List ───────────────────────────────────────────────────────────

    #[test]
    fn cache_kv_namespace_list_lists_per_repo_directory_sizes() {
        let kv = temp_dir("list-per-repo");
        // Two synthetic <fp_short> dirs with different block counts +
        // sizes. Hex0 fanout matches block_path layout (D5).
        touch_block(&kv, "aaaaaaaaaaaaaaaa", "a000000000000000000000000000000000000000000000000000000000000000", &vec![0u8; 1024]);
        touch_block(&kv, "aaaaaaaaaaaaaaaa", "a111111111111111111111111111111111111111111111111111111111111111", &vec![0u8; 2048]);
        touch_block(&kv, "bbbbbbbbbbbbbbbb", "b222222222222222222222222222222222222222222222222222222222222222", &vec![0u8; 4096]);

        let entries = list_namespaces(&kv).unwrap();
        assert_eq!(entries.len(), 2, "two namespaces");
        assert_eq!(entries[0].fp_short, "aaaaaaaaaaaaaaaa");
        assert_eq!(entries[0].block_count, 2);
        assert!(
            entries[0].bytes_on_disk >= 1024 + 2048,
            "bytes_on_disk includes both blocks: {}",
            entries[0].bytes_on_disk
        );
        assert_eq!(entries[1].fp_short, "bbbbbbbbbbbbbbbb");
        assert_eq!(entries[1].block_count, 1);
        assert!(entries[1].bytes_on_disk >= 4096);

        fs::remove_dir_all(&kv).ok();
    }

    #[test]
    fn cache_kv_namespace_list_tolerates_missing_root() {
        let kv = temp_dir("list-missing-root");
        // Don't create models/ subdir.
        let entries = list_namespaces(&kv).unwrap();
        assert!(entries.is_empty());
        fs::remove_dir_all(&kv).ok();
    }

    // ── Size ───────────────────────────────────────────────────────────

    #[test]
    fn cache_kv_namespace_size_returns_zero_on_empty_dir() {
        let kv = temp_dir("size-empty");
        // No models/ subdir at all.
        assert_eq!(total_bytes(&kv), 0);

        // Empty models/ subdir.
        fs::create_dir_all(kv.join("models")).unwrap();
        assert_eq!(total_bytes(&kv), 0);

        fs::remove_dir_all(&kv).ok();
    }

    #[test]
    fn cache_kv_namespace_size_sums_all_namespaces() {
        let kv = temp_dir("size-sum");
        touch_block(
            &kv,
            "aaaaaaaaaaaaaaaa",
            "a000000000000000000000000000000000000000000000000000000000000000",
            &vec![0u8; 1024],
        );
        touch_block(
            &kv,
            "bbbbbbbbbbbbbbbb",
            "b000000000000000000000000000000000000000000000000000000000000000",
            &vec![0u8; 2048],
        );
        let total = total_bytes(&kv);
        assert!(total >= 1024 + 2048, "got {}", total);
        fs::remove_dir_all(&kv).ok();
    }

    // ── Clear ──────────────────────────────────────────────────────────

    #[test]
    fn cache_kv_namespace_clear_removes_only_targeted_repo() {
        let kv = temp_dir("clear-targeted");
        let target_fp = fp_short_for("acme/repo-A", QuantType::Q4_K_M);
        let other_fp = fp_short_for("acme/repo-B", QuantType::Q4_K_M);
        assert_ne!(target_fp, other_fp);

        touch_block(
            &kv,
            &target_fp,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            &vec![0u8; 4096],
        );
        touch_block(
            &kv,
            &other_fp,
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            &vec![0u8; 8192],
        );
        touch_lockdir(&kv, &target_fp);

        let outcome = clear_namespace(&kv, "acme/repo-A", QuantType::Q4_K_M, false).unwrap();
        assert!(outcome.existed);
        assert_eq!(outcome.fp_short, target_fp);
        assert!(outcome.bytes_freed >= 4096);

        // Targeted repo dir gone.
        assert!(
            !kv.join("models").join(&target_fp).exists(),
            "target dir removed"
        );
        // Sibling repo dir survived.
        assert!(
            kv.join("models").join(&other_fp).exists(),
            "sibling repo untouched"
        );
        // locks/ subtree survived (per §11 #4 docstring).
        assert!(
            kv.join("locks").exists(),
            "locks dir not touched by clear --kv-namespace"
        );
        let lock_files: Vec<_> = fs::read_dir(kv.join("locks"))
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(!lock_files.is_empty(), "lock file survived");

        fs::remove_dir_all(&kv).ok();
    }

    #[test]
    fn cache_kv_namespace_clear_is_idempotent_on_missing_repo() {
        let kv = temp_dir("clear-idempotent");
        // No models/ subtree; clear should not error.
        let outcome = clear_namespace(&kv, "never/cached", QuantType::Q8_0, false).unwrap();
        assert!(!outcome.existed);
        assert_eq!(outcome.bytes_freed, 0);
        fs::remove_dir_all(&kv).ok();
    }

    #[test]
    fn cache_kv_namespace_clear_all_quants_removes_every_variant() {
        let kv = temp_dir("clear-all-quants");
        let fp_q8 = fp_short_for("acme/repo-X", QuantType::Q8_0);
        let fp_q4km = fp_short_for("acme/repo-X", QuantType::Q4_K_M);
        let fp_other = fp_short_for("acme/repo-Y", QuantType::Q4_K_M);

        touch_block(
            &kv,
            &fp_q8,
            "1100000000000000000000000000000000000000000000000000000000000000",
            &vec![0u8; 1024],
        );
        touch_block(
            &kv,
            &fp_q4km,
            "2200000000000000000000000000000000000000000000000000000000000000",
            &vec![0u8; 2048],
        );
        touch_block(
            &kv,
            &fp_other,
            "3300000000000000000000000000000000000000000000000000000000000000",
            &vec![0u8; 4096],
        );

        let outcomes = clear_namespace_all_quants(&kv, "acme/repo-X", false).unwrap();
        assert_eq!(outcomes.len(), 4, "one outcome per QuantType variant");
        let removed_count = outcomes.iter().filter(|o| o.existed).count();
        assert_eq!(
            removed_count, 2,
            "both Q8_0 and Q4_K_M existed and were removed"
        );

        assert!(!kv.join("models").join(&fp_q8).exists());
        assert!(!kv.join("models").join(&fp_q4km).exists());
        // Sibling repo untouched.
        assert!(kv.join("models").join(&fp_other).exists());
        fs::remove_dir_all(&kv).ok();
    }

    #[test]
    fn cache_kv_namespace_clear_refuses_when_lock_is_held() {
        let kv = temp_dir("clear-locked");
        let target_fp = fp_short_for("acme/repo-locked", QuantType::Q4_K_M);
        touch_block(
            &kv,
            &target_fp,
            "9900000000000000000000000000000000000000000000000000000000000000",
            &vec![0u8; 1024],
        );

        // Hold the sentinel ourselves to simulate a live serve.
        let _held = active_serve_guard(&kv).expect("hold sentinel");

        let result = clear_namespace(&kv, "acme/repo-locked", QuantType::Q4_K_M, false);
        match result {
            Err(ClearRefusalErr::Locked(msg)) => {
                assert!(
                    msg.contains(".cache_lock"),
                    "diagnostic mentions sentinel: {}",
                    msg
                );
            }
            Err(other) => panic!("expected Locked, got {:?}", other),
            Ok(o) => panic!("expected Locked refusal, got success {:?}", o),
        }
        // Block file survived the refused clear.
        assert!(kv.join("models").join(&target_fp).exists());

        // --force overrides the guard.
        drop(_held);
        // Re-acquire to prove --force still wins under contention.
        let _held2 = active_serve_guard(&kv).expect("reacquire sentinel");
        let outcome =
            clear_namespace(&kv, "acme/repo-locked", QuantType::Q4_K_M, /*force=*/ true)
                .unwrap();
        assert!(outcome.existed);
        assert!(!kv.join("models").join(&target_fp).exists());

        fs::remove_dir_all(&kv).ok();
    }

    // ── Path resolution ────────────────────────────────────────────────

    #[test]
    fn cache_kv_namespace_resolve_kv_root_prefers_explicit_path() {
        let p = PathBuf::from("/tmp/hf2q-explicit-path");
        // Even with the env set, --kv-path wins.
        std::env::set_var(HF2Q_KV_PERSIST_PATH_ENV, "/tmp/hf2q-from-env");
        let resolved = resolve_kv_root(Some(&p)).unwrap();
        std::env::remove_var(HF2Q_KV_PERSIST_PATH_ENV);
        assert_eq!(resolved, p);
    }

    #[test]
    fn cache_kv_namespace_resolve_kv_root_errors_when_neither_set() {
        // Snapshot + clear the env var so the test is hermetic.
        let saved = std::env::var(HF2Q_KV_PERSIST_PATH_ENV).ok();
        std::env::remove_var(HF2Q_KV_PERSIST_PATH_ENV);
        let err = resolve_kv_root(None).unwrap_err();
        if let Some(v) = saved {
            std::env::set_var(HF2Q_KV_PERSIST_PATH_ENV, v);
        }
        let msg = format!("{}", err);
        assert!(
            msg.contains("--kv-path") && msg.contains(HF2Q_KV_PERSIST_PATH_ENV),
            "diagnostic mentions both surfaces: {}",
            msg
        );
    }

    // ── Fingerprint translation ────────────────────────────────────────

    #[test]
    fn cache_kv_namespace_fp_short_matches_spiller_call() {
        // This is the same call the spiller makes
        // (`spiller.rs::family_model_fp` lines 242-244).
        let fp = fp_short_for("google/gemma-4-27b-it", QuantType::Q4_K_M);
        let direct =
            compute_model_fingerprint("google/gemma-4-27b-it", "Q4_K_M", "", "", "").short_hex();
        assert_eq!(fp, direct);
        assert_eq!(fp.len(), 16, "16 hex chars per format.rs:173-175");
    }
}
