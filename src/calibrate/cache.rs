//! Sensitivity-JSON cache (ADR-014 P5 Decision 9 / item #5).
//!
//! ## Why this exists
//!
//! DWQ calibration runs a forward pass through the calibration corpus
//! to capture activations and produce per-layer sensitivity scores.
//! Different bit-pair variants of the **same model on the same corpus**
//! (e.g. `dwq-4-6` and `dwq-4-8`) produce identical sensitivity scores
//! — only the downstream bit-allocation table differs. Without caching,
//! each `hf2q convert ... --quant dwq-X-Y` invocation re-runs the
//! forward pass from scratch, wasting tens of minutes per variant.
//!
//! This cache stores the per-layer sensitivity vector keyed on the
//! triple `(model_sha, corpus_sha, sensitivity_algorithm_version)`. The
//! second `hf2q convert ... --quant dwq-4-8` (after a `dwq-4-6` run on
//! the same model + same corpus) gets a cache hit and skips the
//! forward pass entirely.
//!
//! ## Cache layout
//!
//! Files live at `${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/sensitivity/<key>.json`.
//! `<key>` is the hex-encoded SHA-256 of the canonicalised triple. JSON
//! payload contains the sensitivity vector + the inputs used to derive
//! the key (for diagnostics / version mismatch detection). Stale
//! entries (e.g. when [`SENSITIVITY_ALGORITHM_VERSION`] is bumped) miss
//! by key construction; a manual purge of `~/.cache/hf2q/sensitivity/`
//! is the recovery path. The cache is opt-in: callers explicitly
//! [`load`] before forward pass and [`save`] after.
//!
//! ## Sovereignty
//!
//! Pure Rust. `sha2` + `hex` runtime deps already in Cargo.toml. No
//! `dirs` crate — `~/.cache` resolution uses `$XDG_CACHE_HOME` then
//! `$HOME` from `std::env::var`. P7 will reuse this cache module when
//! the calibrator orchestration moves to `src/calibrate/`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Pinned version string for the sensitivity computation algorithm.
/// Bump when [`crate::calibrate::sensitivity::compute_layer_sensitivity`]
/// or any of its inputs (variance/max-magnitude formula, prefilter
/// rules, layer enumeration order) changes — the bump invalidates
/// every previously-cached entry by changing the key derivation.
///
/// Format: `<major>.<minor>.<algo>` — `algo` is a free-text tag for
/// the human-readable algorithm name. Examples:
/// - `1.0.variance-magnitude`: original
///   `sqrt(variance) * log2(1 + max_magnitude)` from
///   `src/quantize/sensitivity.rs`.
pub const SENSITIVITY_ALGORITHM_VERSION: &str = "1.0.variance-magnitude";

/// Errors from cache I/O and key derivation.
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("sensitivity cache: I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("sensitivity cache: JSON serialise/deserialise failed at {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("sensitivity cache: cannot resolve cache directory ($XDG_CACHE_HOME and $HOME both unset)")]
    NoHome,

    #[error("sensitivity cache: hit on key {key} but algorithm version mismatch (cached {cached}, current {current})")]
    AlgorithmVersionMismatch {
        key: String,
        cached: String,
        current: String,
    },
}

/// Inputs that combine to produce a cache key. Each must be a
/// deterministic content hash — non-deterministic inputs (e.g. file
/// modification times, random nonces) would produce false misses on
/// every run.
#[derive(Debug, Clone)]
pub struct SensitivityCacheKey {
    /// SHA-256 of the model's safetensors bytes (or, for cache use,
    /// any deterministic per-model content hash). Callers typically
    /// pass the SHA from `tokenizer.json` or a hashed concatenation
    /// of the model's tensor data — anything that uniquely identifies
    /// "the model whose activations we're calibrating".
    pub model_sha: String,

    /// SHA-256 of the calibration corpus tokens. Different corpora
    /// produce different sensitivity vectors; the key disambiguates.
    pub corpus_sha: String,

    /// Sensitivity algorithm version. Defaults to
    /// [`SENSITIVITY_ALGORITHM_VERSION`]; callers can override for
    /// future-algorithm experiments.
    pub algorithm_version: String,
}

impl SensitivityCacheKey {
    /// Construct a cache key with the current algorithm version.
    pub fn new(model_sha: impl Into<String>, corpus_sha: impl Into<String>) -> Self {
        Self {
            model_sha: model_sha.into(),
            corpus_sha: corpus_sha.into(),
            algorithm_version: SENSITIVITY_ALGORITHM_VERSION.to_string(),
        }
    }

    /// Construct a cache key with an explicit algorithm version (for
    /// experiments or replay against historical caches).
    pub fn with_algorithm_version(
        model_sha: impl Into<String>,
        corpus_sha: impl Into<String>,
        algorithm_version: impl Into<String>,
    ) -> Self {
        Self {
            model_sha: model_sha.into(),
            corpus_sha: corpus_sha.into(),
            algorithm_version: algorithm_version.into(),
        }
    }

    /// Compute the hex-encoded SHA-256 hash of this key's canonical
    /// triple `model_sha|corpus_sha|algorithm_version`. The pipe
    /// separator avoids ambiguity from concatenated hex strings of
    /// arbitrary length.
    pub fn hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.model_sha.as_bytes());
        h.update(b"|");
        h.update(self.corpus_sha.as_bytes());
        h.update(b"|");
        h.update(self.algorithm_version.as_bytes());
        hex::encode(h.finalize())
    }
}

/// On-disk JSON payload. Carries the sensitivity vector plus the
/// inputs that produced the key so a diagnostic dump can verify the
/// expected algorithm version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityCacheEntry {
    /// Algorithm version that produced this entry — used at load time
    /// to fail loudly if a stale cache file lingers from a previous
    /// algorithm.
    pub algorithm_version: String,

    /// Inputs that produced the key — informational, not used for
    /// lookup (the key is the filename).
    pub model_sha: String,
    pub corpus_sha: String,

    /// Per-layer sensitivity vector. Mirrors the public shape of
    /// [`crate::calibrate::sensitivity::LayerSensitivity`] for direct
    /// round-trip; the JSON field names match. Stored here as plain
    /// f64 fields for serde-friendliness without depending on the
    /// `LayerSensitivity` type (which lives in src/quantize/ until P7).
    pub layers: Vec<CachedLayerSensitivity>,
}

/// Serde-friendly mirror of `LayerSensitivity` (lives in
/// `src/quantize/sensitivity.rs` until P7's path migration). When P7
/// moves sensitivity.rs into `src/calibrate/`, this type collapses
/// into a `From<LayerSensitivity>` adapter or is replaced outright.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedLayerSensitivity {
    pub layer_index: usize,
    pub variance: f64,
    pub max_magnitude: f64,
    pub score: f64,
}

/// Resolve the sensitivity cache directory, creating it if necessary.
///
/// Honours `$XDG_CACHE_HOME` (Linux convention) before falling back to
/// `$HOME/.cache/`. Returns the absolute path to
/// `<cache_root>/hf2q/sensitivity/`.
pub fn cache_dir() -> Result<PathBuf, CacheError> {
    let cache_root = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            PathBuf::from(xdg)
        } else {
            home_dot_cache()?
        }
    } else {
        home_dot_cache()?
    };
    let dir = cache_root.join("hf2q").join("sensitivity");
    fs::create_dir_all(&dir).map_err(|e| CacheError::Io {
        path: dir.display().to_string(),
        source: e,
    })?;
    Ok(dir)
}

fn home_dot_cache() -> Result<PathBuf, CacheError> {
    let home = std::env::var("HOME").map_err(|_| CacheError::NoHome)?;
    if home.is_empty() {
        return Err(CacheError::NoHome);
    }
    Ok(PathBuf::from(home).join(".cache"))
}

/// Compute the on-disk cache file path for a given key.
pub fn cache_file_path(key: &SensitivityCacheKey) -> Result<PathBuf, CacheError> {
    Ok(cache_dir()?.join(format!("{}.json", key.hash())))
}

/// Save a sensitivity vector under the given cache key. Atomic via
/// write-to-temp + rename to avoid leaving truncated files on disk if
/// the process is interrupted mid-write.
pub fn save(
    key: &SensitivityCacheKey,
    layers: &[CachedLayerSensitivity],
) -> Result<(), CacheError> {
    let path = cache_file_path(key)?;
    save_to_path(&path, key, layers)
}

/// Save under an explicit path (used by tests with a tempdir; the
/// public [`save`] wraps this with the canonical cache location).
pub fn save_to_path(
    path: &Path,
    key: &SensitivityCacheKey,
    layers: &[CachedLayerSensitivity],
) -> Result<(), CacheError> {
    let entry = SensitivityCacheEntry {
        algorithm_version: key.algorithm_version.clone(),
        model_sha: key.model_sha.clone(),
        corpus_sha: key.corpus_sha.clone(),
        layers: layers.to_vec(),
    };
    let serialised = serde_json::to_string_pretty(&entry).map_err(|e| CacheError::Json {
        path: path.display().to_string(),
        source: e,
    })?;

    // Atomic write: temp file + rename. POSIX rename is atomic; Ctrl+C
    // mid-write leaves the tempfile (cleaned up by parent dir purge)
    // but never a half-written canonical file.
    let parent = path.parent().ok_or_else(|| CacheError::Io {
        path: path.display().to_string(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "no parent dir"),
    })?;
    fs::create_dir_all(parent).map_err(|e| CacheError::Io {
        path: parent.display().to_string(),
        source: e,
    })?;

    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, serialised.as_bytes()).map_err(|e| CacheError::Io {
        path: tmp_path.display().to_string(),
        source: e,
    })?;
    fs::rename(&tmp_path, path).map_err(|e| CacheError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(())
}

/// Look up a cached sensitivity vector for the given key.
///
/// Returns:
/// - `Ok(Some(layers))` on cache hit with matching algorithm version.
/// - `Ok(None)` on cache miss (file does not exist).
/// - `Err(AlgorithmVersionMismatch)` if a file exists at the canonical
///   path but its `algorithm_version` field disagrees with
///   `key.algorithm_version` — unusual (since the algorithm version is
///   part of the key derivation), but possible if a caller wrote with
///   one version and reads with another.
pub fn load(
    key: &SensitivityCacheKey,
) -> Result<Option<Vec<CachedLayerSensitivity>>, CacheError> {
    let path = cache_file_path(key)?;
    load_from_path(&path, key)
}

/// Load from an explicit path (used by tests).
pub fn load_from_path(
    path: &Path,
    key: &SensitivityCacheKey,
) -> Result<Option<Vec<CachedLayerSensitivity>>, CacheError> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path).map_err(|e| CacheError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let entry: SensitivityCacheEntry =
        serde_json::from_str(&raw).map_err(|e| CacheError::Json {
            path: path.display().to_string(),
            source: e,
        })?;

    if entry.algorithm_version != key.algorithm_version {
        return Err(CacheError::AlgorithmVersionMismatch {
            key: key.hash(),
            cached: entry.algorithm_version,
            current: key.algorithm_version.clone(),
        });
    }
    Ok(Some(entry.layers))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_layers() -> Vec<CachedLayerSensitivity> {
        vec![
            CachedLayerSensitivity {
                layer_index: 0,
                variance: 0.5,
                max_magnitude: 2.0,
                score: 0.707 * 1.585,
            },
            CachedLayerSensitivity {
                layer_index: 1,
                variance: 1.0,
                max_magnitude: 3.0,
                score: 1.0 * 2.0,
            },
        ]
    }

    /// Same inputs produce the same hex-encoded hash.
    #[test]
    fn cache_key_determinism() {
        let k1 = SensitivityCacheKey::new("model-abc", "corpus-xyz");
        let k2 = SensitivityCacheKey::new("model-abc", "corpus-xyz");
        assert_eq!(k1.hash(), k2.hash());
        // hash is 64-char hex (SHA-256 → 32 bytes → 64 hex chars).
        assert_eq!(k1.hash().len(), 64);
        assert!(k1.hash().chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// Different inputs produce different hashes (no collision on the
    /// minimal disambiguators).
    #[test]
    fn cache_key_disjoint() {
        let k_model_a = SensitivityCacheKey::new("model-A", "corpus");
        let k_model_b = SensitivityCacheKey::new("model-B", "corpus");
        assert_ne!(k_model_a.hash(), k_model_b.hash());

        let k_corpus_a = SensitivityCacheKey::new("model", "corpus-A");
        let k_corpus_b = SensitivityCacheKey::new("model", "corpus-B");
        assert_ne!(k_corpus_a.hash(), k_corpus_b.hash());

        let k_alg_a = SensitivityCacheKey::with_algorithm_version("model", "corpus", "1.0");
        let k_alg_b = SensitivityCacheKey::with_algorithm_version("model", "corpus", "2.0");
        assert_ne!(k_alg_a.hash(), k_alg_b.hash());
    }

    /// Pipe separator avoids the
    ///   model="ab", corpus="c"  ↔  model="a", corpus="bc"
    /// collision that plain concatenation would produce.
    #[test]
    fn cache_key_pipe_separator_avoids_concat_collision() {
        let k1 = SensitivityCacheKey::with_algorithm_version("ab", "c", "v");
        let k2 = SensitivityCacheKey::with_algorithm_version("a", "bc", "v");
        assert_ne!(
            k1.hash(),
            k2.hash(),
            "pipe-separated digest must distinguish 'ab|c' from 'a|bc'"
        );
    }

    /// Save → load round-trip reproduces the same data.
    #[test]
    fn cache_save_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let key = SensitivityCacheKey::new("model-1", "corpus-1");
        let path = tmp.path().join(format!("{}.json", key.hash()));

        let layers = fixture_layers();
        save_to_path(&path, &key, &layers).unwrap();

        let loaded = load_from_path(&path, &key).unwrap();
        assert_eq!(loaded, Some(layers));
    }

    /// Cache miss returns `Ok(None)`, not an error.
    #[test]
    fn cache_load_miss_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let key = SensitivityCacheKey::new("model-1", "corpus-1");
        let path = tmp.path().join(format!("{}.json", key.hash()));
        let loaded = load_from_path(&path, &key).unwrap();
        assert!(loaded.is_none());
    }

    /// Algorithm version mismatch surfaces a typed error (not silent
    /// stale-data return).
    #[test]
    fn cache_algorithm_version_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let key_v1 =
            SensitivityCacheKey::with_algorithm_version("model", "corpus", "v1.alpha");
        let path = tmp.path().join("entry.json");
        save_to_path(&path, &key_v1, &fixture_layers()).unwrap();

        let key_v2 =
            SensitivityCacheKey::with_algorithm_version("model", "corpus", "v2.beta");
        let result = load_from_path(&path, &key_v2);
        match result {
            Err(CacheError::AlgorithmVersionMismatch {
                cached, current, ..
            }) => {
                assert_eq!(cached, "v1.alpha");
                assert_eq!(current, "v2.beta");
            }
            _ => panic!("expected AlgorithmVersionMismatch, got {result:?}"),
        }
    }

    /// Atomic write: rename pattern leaves no partial file at the
    /// canonical path even if the writer is interrupted (simulated by
    /// inspecting that no `.tmp` file lingers after a successful save).
    #[test]
    fn cache_save_atomic_no_tmp_lingers() {
        let tmp = tempfile::tempdir().unwrap();
        let key = SensitivityCacheKey::new("model", "corpus");
        let path = tmp.path().join(format!("{}.json", key.hash()));
        save_to_path(&path, &key, &fixture_layers()).unwrap();

        // Canonical file exists; no .tmp lingering.
        assert!(path.exists());
        let tmp_path = path.with_extension("json.tmp");
        assert!(!tmp_path.exists(), "tempfile must be renamed away");
    }

    // Note: `cache_dir()`'s `$XDG_CACHE_HOME → $HOME/.cache` resolution
    // is deliberately NOT tested with env-var mutation because the
    // crate's existing `arch::smoke::tests::hf_token_falls_back_*`
    // tests mutate `HOME` / `HF_TOKEN` via a serialising `env_lock`,
    // and a parallel-running env-mutating test here would race even
    // if it took the same lock (the lock is private to `arch::smoke`).
    // The resolution logic is simple enough that visual review suffices;
    // the integration tests in `tests/convert_qwen35_*.rs` exercise the
    // sensitivity cache via the full convert pipeline at end-to-end
    // scope, providing real-world coverage without env-mutation
    // contention.

    /// Empty layers vector round-trips correctly (no panics on empty).
    #[test]
    fn cache_save_load_empty_layers() {
        let tmp = tempfile::tempdir().unwrap();
        let key = SensitivityCacheKey::new("model", "corpus");
        let path = tmp.path().join("empty.json");
        save_to_path(&path, &key, &[]).unwrap();
        let loaded = load_from_path(&path, &key).unwrap().unwrap();
        assert!(loaded.is_empty());
    }
}
