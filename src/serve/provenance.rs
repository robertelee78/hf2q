//! ADR-005 Phase 4 iter-207 — GGUF provenance reader.
//!
//! Detects whether an opened GGUF was produced by hf2q itself by reading
//! three optional metadata keys from the GGUF KV section.  When present,
//! the auto-pipeline ([`crate::serve::auto_pipeline`]) cross-verifies the
//! GGUF's claimed source-bundle SHA-256 against the cache manifest's
//! recorded `source_shards` (via
//! [`crate::serve::cache::compute_source_bundle_sha256`]) and short-
//! circuits the per-load 30 GB SHA-256 integrity re-check.  External
//! GGUFs (llama.cpp's official quants, anything we didn't emit) carry
//! none of the keys and continue to run [`crate::serve::cache::ModelCache::verify_quantized`]
//! exactly as W71 / iter-203 designed.
//!
//! # Schema (three keys)
//!
//! ```text
//! hf2q.producer_version  String  REQUIRED  e.g. "hf2q 0.1.0" — `env!("CARGO_PKG_VERSION")` prefixed.
//! hf2q.source_sha256     String  REQUIRED  lowercase-hex SHA-256 of the canonical source-shard manifest
//!                                          (algo: [`crate::serve::cache::compute_source_bundle_sha256`]).
//! hf2q.mmproj_sha256     String  OPTIONAL  lowercase-hex SHA-256 of the paired vision projector GGUF, if any.
//! ```
//!
//! Both required keys must be present + non-empty for the reader to
//! return `Provenance::Hf2q`; any other shape (one missing, both missing,
//! either empty) is treated as `External`.  This is intentionally
//! conservative — a half-stamped GGUF is treated like an external one,
//! never short-circuiting based on incomplete provenance.
//!
//! # Writer side (DEFERRED — ADR-014 P7 fence)
//!
//! As of iter-207 no hf2q-emitted GGUF carries these keys (the writer
//! lives in `src/backends/gguf.rs`, fenced by ADR-014 P7).  Reader-side
//! ships clean today: `detect` returns `External` for every existing
//! GGUF, the auto-pipeline runs `verify_quantized` exactly as before,
//! no behaviour change.  Once ADR-014 P7 closes, a fast-follow writer
//! iter (iter-208/9 or later) emits the keys at quantize time, and the
//! short-circuit activates naturally on the next cache hit.
//!
//! # Hex normalization
//!
//! SHA-256 strings are accepted in either case and lowercased before
//! comparison so a writer that emits uppercase hex (or mixed case) will
//! still match the cache's lowercase-hex bundle SHA.  This guards
//! against a class of cross-version regressions where the writer is
//! upgraded independently of the reader.

use mlx_native::gguf::GgufFile;

// ---------------------------------------------------------------------
// Public schema
// ---------------------------------------------------------------------

/// Canonical GGUF metadata key for the producer version stamp.  Mirrors
/// `hf2q.producer_version` on disk.  Public so writer-side code (once
/// the ADR-014 P7 fence releases) can use the same constant.
pub const KEY_PRODUCER_VERSION: &str = "hf2q.producer_version";

/// Canonical GGUF metadata key for the source-bundle SHA-256.  Value is
/// the hex digest computed by
/// [`crate::serve::cache::compute_source_bundle_sha256`].
pub const KEY_SOURCE_SHA256: &str = "hf2q.source_sha256";

/// Canonical GGUF metadata key for the paired mmproj SHA-256.  Optional;
/// only emitted by the writer for vision-paired models.
pub const KEY_MMPROJ_SHA256: &str = "hf2q.mmproj_sha256";

/// Result of inspecting a GGUF's metadata for hf2q-origin provenance.
///
/// A GGUF is classified `Hf2q` only when both required keys are present
/// and non-empty; everything else is `External`.  Down-stream short-
/// circuit logic must additionally cross-verify
/// [`Hf2q::source_sha256`] against the cache's
/// [`crate::serve::cache::compute_source_bundle_sha256`] before
/// trusting the provenance — a hf2q-stamped GGUF whose claimed source
/// SHA does NOT match the cache's recorded shards is treated as an
/// integrity error, not a short-circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provenance {
    /// Both required keys read successfully.
    Hf2q {
        /// Verbatim contents of `hf2q.producer_version`.  Free-form
        /// string — the canonical writer emits `"hf2q <semver>"` but
        /// other formats are accepted (the reader does no parsing
        /// beyond presence + non-empty).
        producer_version: String,
        /// Lowercase-hex SHA-256 of the source bundle (normalized from
        /// the on-disk value, which may be upper / mixed case).
        source_sha256: String,
        /// `Some(_)` iff `hf2q.mmproj_sha256` was present and non-empty;
        /// `None` for non-vision models or vision models that omit the
        /// pairing.  Lowercase-hex.
        mmproj_sha256: Option<String>,
    },
    /// At least one required key is missing or empty — treat as a
    /// non-hf2q GGUF and run the full integrity check.
    External,
}

impl Provenance {
    /// Convenience: returns `true` only for the `Hf2q` variant.
    pub fn is_hf2q(&self) -> bool {
        matches!(self, Provenance::Hf2q { .. })
    }
}

// ---------------------------------------------------------------------
// Trait abstraction over the metadata read API
// ---------------------------------------------------------------------

/// Minimal lookup surface needed by [`detect`].
///
/// Production code passes a [`GgufFile`] (whose
/// [`metadata_string`](GgufFile::metadata_string) implements this); unit
/// tests pass a `HashMap<String, String>` so the provenance schema can
/// be exercised without standing up a full GGUF parser fixture.
///
/// Restricted to string keys on purpose — every key in the provenance
/// schema is a string, and the trait deliberately won't grow into a
/// full metadata-introspection surface (that's `GgufFile`'s job).
pub trait MetadataLookup {
    fn get_string(&self, key: &str) -> Option<&str>;
}

impl MetadataLookup for GgufFile {
    fn get_string(&self, key: &str) -> Option<&str> {
        self.metadata_string(key)
    }
}

// `HashMap<String, String>` blanket impl — used by unit tests below to
// build fixture metadata without touching GGUF parsing.  Generic over
// the hasher so callers can use either `std::collections::HashMap` or
// hashbrown's variant transparently.
impl<S: std::hash::BuildHasher> MetadataLookup for std::collections::HashMap<String, String, S> {
    fn get_string(&self, key: &str) -> Option<&str> {
        self.get(key).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------

/// Read the three provenance keys from `metadata` and classify.
///
/// Returns:
/// - [`Provenance::Hf2q`] when both [`KEY_PRODUCER_VERSION`] and
///   [`KEY_SOURCE_SHA256`] are present and non-empty.
/// - [`Provenance::External`] in every other case (either missing, empty
///   string, partial schema, or wrong type).
///
/// SHA-256 strings are lowercased before being stored in the returned
/// [`Provenance::Hf2q`] so downstream comparison against the cache's
/// `compute_source_bundle_sha256` (also lowercase) is byte-equal.
pub fn detect<M: MetadataLookup + ?Sized>(metadata: &M) -> Provenance {
    let producer_version = match metadata.get_string(KEY_PRODUCER_VERSION) {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => return Provenance::External,
    };
    let source_sha256 = match metadata.get_string(KEY_SOURCE_SHA256) {
        Some(v) if !v.is_empty() => v.to_ascii_lowercase(),
        _ => return Provenance::External,
    };
    // Optional: present-and-non-empty → Some(lowercase); absent or
    // empty → None.  An empty string is treated as "not present" so a
    // writer that emits `""` for non-vision models doesn't fool the
    // reader into thinking there's a paired projector.
    let mmproj_sha256 = metadata
        .get_string(KEY_MMPROJ_SHA256)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_ascii_lowercase());

    Provenance::Hf2q {
        producer_version,
        source_sha256,
        mmproj_sha256,
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    // ── Required-key presence ───────────────────────────────────────────

    #[test]
    fn detect_returns_hf2q_when_required_keys_present() {
        let m = meta(&[
            (KEY_PRODUCER_VERSION, "hf2q 0.1.0"),
            (KEY_SOURCE_SHA256, &"a".repeat(64)),
        ]);
        match detect(&m) {
            Provenance::Hf2q {
                producer_version,
                source_sha256,
                mmproj_sha256,
            } => {
                assert_eq!(producer_version, "hf2q 0.1.0");
                assert_eq!(source_sha256, "a".repeat(64));
                assert_eq!(mmproj_sha256, None);
            }
            other => panic!("expected Hf2q, got {other:?}"),
        }
    }

    #[test]
    fn detect_returns_external_when_keys_absent() {
        let m = meta(&[
            ("general.architecture", "qwen35"),
            ("general.name", "Qwen3.5-MoE"),
        ]);
        assert_eq!(detect(&m), Provenance::External);
        assert!(!detect(&m).is_hf2q());
    }

    #[test]
    fn detect_returns_external_when_partial_keys_present() {
        // Only producer_version, no source_sha256.
        let only_version = meta(&[(KEY_PRODUCER_VERSION, "hf2q 0.1.0")]);
        assert_eq!(detect(&only_version), Provenance::External);

        // Only source_sha256, no producer_version.
        let only_sha = meta(&[(KEY_SOURCE_SHA256, &"a".repeat(64))]);
        assert_eq!(detect(&only_sha), Provenance::External);

        // mmproj_sha256 alone is never enough.
        let only_mmproj = meta(&[(KEY_MMPROJ_SHA256, &"b".repeat(64))]);
        assert_eq!(detect(&only_mmproj), Provenance::External);
    }

    #[test]
    fn detect_returns_external_when_required_key_is_empty_string() {
        // Empty producer_version → External.
        let empty_version = meta(&[
            (KEY_PRODUCER_VERSION, ""),
            (KEY_SOURCE_SHA256, &"a".repeat(64)),
        ]);
        assert_eq!(detect(&empty_version), Provenance::External);

        // Empty source_sha256 → External.
        let empty_sha = meta(&[(KEY_PRODUCER_VERSION, "hf2q 0.1.0"), (KEY_SOURCE_SHA256, "")]);
        assert_eq!(detect(&empty_sha), Provenance::External);
    }

    #[test]
    fn detect_extracts_mmproj_sha256_when_present() {
        let m = meta(&[
            (KEY_PRODUCER_VERSION, "hf2q 0.1.0"),
            (KEY_SOURCE_SHA256, &"c".repeat(64)),
            (KEY_MMPROJ_SHA256, &"d".repeat(64)),
        ]);
        match detect(&m) {
            Provenance::Hf2q { mmproj_sha256, .. } => {
                assert_eq!(mmproj_sha256.as_deref(), Some("d".repeat(64).as_str()));
            }
            other => panic!("expected Hf2q with mmproj, got {other:?}"),
        }
    }

    #[test]
    fn detect_treats_empty_mmproj_as_absent() {
        // A writer that emits `""` for non-vision models must not be
        // mistaken for a vision pairing.
        let m = meta(&[
            (KEY_PRODUCER_VERSION, "hf2q 0.1.0"),
            (KEY_SOURCE_SHA256, &"e".repeat(64)),
            (KEY_MMPROJ_SHA256, ""),
        ]);
        match detect(&m) {
            Provenance::Hf2q { mmproj_sha256, .. } => {
                assert_eq!(mmproj_sha256, None);
            }
            other => panic!("expected Hf2q, got {other:?}"),
        }
    }

    // ── Defensive normalization ─────────────────────────────────────────

    #[test]
    fn detect_handles_unicode_in_producer_version() {
        // Defensive: producer_version is free-form; emoji + multibyte
        // shouldn't break the read.
        let banner = "hf2q 0.1.0 \u{1F4DA}\u{1F680} test-build";
        let m = meta(&[
            (KEY_PRODUCER_VERSION, banner),
            (KEY_SOURCE_SHA256, &"f".repeat(64)),
        ]);
        match detect(&m) {
            Provenance::Hf2q {
                producer_version, ..
            } => {
                assert_eq!(producer_version, banner);
            }
            other => panic!("expected Hf2q, got {other:?}"),
        }
    }

    #[test]
    fn detect_handles_uppercase_hex_in_sha256() {
        // Some writers emit uppercase hex.  Reader must normalize to
        // lowercase so the cross-verify against the cache's lowercase
        // bundle SHA succeeds.
        let upper = "A".repeat(64);
        let m = meta(&[
            (KEY_PRODUCER_VERSION, "hf2q 0.1.0"),
            (KEY_SOURCE_SHA256, &upper),
            (KEY_MMPROJ_SHA256, &"B".repeat(64)),
        ]);
        match detect(&m) {
            Provenance::Hf2q {
                source_sha256,
                mmproj_sha256,
                ..
            } => {
                assert_eq!(source_sha256, "a".repeat(64));
                assert_eq!(mmproj_sha256.as_deref(), Some("b".repeat(64).as_str()));
            }
            other => panic!("expected Hf2q, got {other:?}"),
        }
    }

    #[test]
    fn detect_handles_mixed_case_hex_in_sha256() {
        // Belt-and-braces: 64 chars, mixed case.
        let mixed: String = (0..64)
            .map(|i| if i % 2 == 0 { 'A' } else { 'b' })
            .collect();
        let m = meta(&[
            (KEY_PRODUCER_VERSION, "hf2q 0.1.0"),
            (KEY_SOURCE_SHA256, &mixed),
        ]);
        match detect(&m) {
            Provenance::Hf2q { source_sha256, .. } => {
                assert!(source_sha256.chars().all(|c| !c.is_ascii_uppercase()));
                assert_eq!(source_sha256, mixed.to_ascii_lowercase());
            }
            other => panic!("expected Hf2q, got {other:?}"),
        }
    }

    // ── Constant stability ─────────────────────────────────────────────

    #[test]
    fn metadata_keys_are_namespaced_under_hf2q() {
        // Belt-and-braces guard against accidental rename — these keys
        // are part of the GGUF on-disk schema and must not change
        // without a coordinated reader+writer migration.
        assert_eq!(KEY_PRODUCER_VERSION, "hf2q.producer_version");
        assert_eq!(KEY_SOURCE_SHA256, "hf2q.source_sha256");
        assert_eq!(KEY_MMPROJ_SHA256, "hf2q.mmproj_sha256");
    }
}
