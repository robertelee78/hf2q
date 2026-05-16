//! `SourceShard` + `compute_source_bundle_sha256` — the canonical
//! source-bundle hash inputs for GGUF provenance binding.
//!
//! Migrated 2026-05-16 from `src/serve/cache.rs` as part of the v0.1.0
//! workspace split (B1.3b).  Migration was gated on B1.3a (integrity
//! types into core) because `SourceShard::from_integrity` needs
//! `crate::core::integrity::ShardIntegrity`.
//!
//! Now that both types are in core, the cache (in serve/) and the
//! GGUF writer (in backends/, the convert side) can share the
//! provenance-binding contract without depending on each other's
//! crate-level modules.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::SystemTime;

use crate::core::integrity::ShardIntegrity;

/// One per-shard integrity record persisted into the cache manifest.
/// Mirrors [`crate::core::integrity::ShardIntegrity`] in JSON shape so
/// the two surfaces stay aligned and the `from_integrity` adapter is a
/// memberwise copy + a `verified_at_secs` stamp.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceShard {
    pub filename: String,
    pub bytes: u64,
    /// Lowercase hex SHA-256 from HF's `x-linked-etag` (LFS-managed
    /// shards only).  `None` for non-LFS files (config.json,
    /// tokenizer.json) — see [`crate::core::integrity`] module docs.
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
    pub fn from_integrity(value: &ShardIntegrity) -> Self {
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

fn secs_since_epoch() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Compute a deterministic source-bundle SHA-256 from a list of source
/// shards — ADR-005 Phase 4 iter-207 (provenance reader).
///
/// The result is the lowercase-hex SHA-256 of `"<filename>:<sha256-hex>\n"`
/// for every shard whose `sha256` is `Some(_)`, sorted by filename.  This
/// is the canonical "source-bundle hash" that a hf2q-emitted GGUF carries
/// as `hf2q.source_sha256` (writer side schedules behind ADR-014 P7).
/// The reader cross-verifies the GGUF's claim against this computed
/// value so a hf2q-origin GGUF can short-circuit the per-load 30 GB
/// integrity re-check.
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn source_shard_adapter_copies_all_fields_and_stamps_timestamp() {
        let integ = ShardIntegrity {
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
    }
}
