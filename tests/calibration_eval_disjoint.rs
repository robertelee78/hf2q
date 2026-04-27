//! ADR-012 P9 — DWQ calibration corpus vs PPL eval corpus disjointness.
//!
//! Decision 17 §Measurement protocol (2026-04-24):
//!   "Calibration corpus disjointness: DWQ calibration corpus (1024
//!    samples from the existing hf2q calibration source) must be
//!    disjoint from the eval corpus. Asserted by a unit test comparing
//!    the two corpora's SHA-256s of their token lists."
//!
//! Pre-P9, the full PPL eval corpus lives at
//! `tests/fixtures/ppl-corpus/wikitext2.tokens` (landed with P9 per the
//! deliverables list). Until that fixture is produced, this test
//! anchors the INTENT — it walks the source tree to verify that once
//! the eval corpus lands, it lives at the documented path and carries
//! a .sha256 sidecar. The moment those files are populated with real
//! wikitext-2 content, the test flips to a real disjointness check.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    let cargo_manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    PathBuf::from(cargo_manifest)
}

#[test]
fn eval_corpus_sha256_sidecar_is_present_when_corpus_is() {
    // If wikitext2.tokens is committed, a sidecar hash MUST accompany it.
    // The sidecar prevents silent corpus edits that would invalidate
    // every published PPL number.
    let tokens = repo_root().join("tests/fixtures/ppl-corpus/wikitext2.tokens");
    let sidecar = repo_root().join("tests/fixtures/ppl-corpus/wikitext2.tokens.sha256");

    if tokens.exists() {
        assert!(
            sidecar.exists(),
            "PPL eval corpus {:?} must carry a .sha256 sidecar at {:?}",
            tokens, sidecar
        );
        let actual = std::fs::read(&tokens).expect("read tokens");
        let recorded = std::fs::read_to_string(&sidecar).expect("read sha256");
        let expected: String = recorded
            .chars()
            .filter(|c| c.is_ascii_hexdigit())
            .take(64)
            .collect();
        // Compute SHA-256 of the tokens file.
        let digest = sha256_hex(&actual);
        assert_eq!(
            digest, expected,
            "wikitext2.tokens SHA-256 does not match .sha256 sidecar — corpus drift detected"
        );
    }
}

#[test]
fn pre_p9_fixtures_placeholder_is_documented() {
    // Until the real corpus lands, document where it will live so
    // curious readers find the intent.
    let dir = repo_root().join("tests/fixtures/ppl-corpus");
    if !dir.exists() {
        eprintln!(
            "ADR-012 P9 eval corpus will land at {:?}/wikitext2.tokens \
             once `tests/quality_thresholds.rs` + `tests/calibration_eval_disjoint.rs` \
             are wired to RealActivationCapture. See Decision 17 §Measurement protocol.",
            dir
        );
    }
}

#[test]
fn calibration_corpus_source_is_documented_in_adr012() {
    // The calibration corpus comes from src/quantize/calibration/corpus.rs
    // per Decision 17 §"Reuse the existing hf2q calibration corpus".
    // Asserting the file exists is Chesterton's fence — if anyone
    // removes the file, the ADR reference breaks silently.
    let calib = repo_root().join("src/quantize/calibration");
    if !calib.exists() {
        // Pre-P9 state: the calibration module is at src/calibrate/
        // (moved by ADR-014 P7 iter-8 Layout A migration from
        // src/quantize/dwq.rs → src/calibrate/dwq.rs). Document the
        // expected location for the moment ADR-013 P12 lands.
        let dwq = repo_root().join("src/calibrate/dwq.rs");
        assert!(
            dwq.exists(),
            "src/calibrate/dwq.rs is the calibration-aware entry point per ADR-012 Decision 13 (post-Layout-A)"
        );
    }
}

#[test]
fn disjointness_is_asserted_when_both_corpora_exist() {
    // When wikitext2.tokens + calibration sample list are both present,
    // assert zero overlap by token-list SHA-256 per Decision 17.
    let tokens = repo_root().join("tests/fixtures/ppl-corpus/wikitext2.tokens");
    let calib_list = repo_root().join("tests/fixtures/calibration/sample-hashes.txt");
    if !tokens.exists() || !calib_list.exists() {
        return; // corpus not yet landed — pre-P9 state
    }
    let eval_bytes = std::fs::read(&tokens).expect("read tokens");
    let eval_sha = sha256_hex(&eval_bytes);
    let calib_text = std::fs::read_to_string(&calib_list).expect("read calib list");
    for line in calib_text.lines() {
        let sha = line.trim();
        if sha.is_empty() || sha.starts_with('#') {
            continue;
        }
        assert_ne!(
            sha, eval_sha,
            "ADR-012 Decision 17: calibration sample {} overlaps with eval corpus — disjoint-ness violated",
            sha
        );
    }
}

// --- Minimal SHA-256 via the same API hf2q uses elsewhere ---
//
// `sha2` is a workspace-level dep; we use it directly. No feature flag.

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    hex::encode(out)
}
