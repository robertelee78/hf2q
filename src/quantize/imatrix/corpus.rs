//! Calibration corpus loader for the imatrix subsystem.
//!
//! Per ADR-033 §Pi: three corpus sources are surface-level supported.
//!
//! * [`CorpusSource::Cdv3`] — bartowski's `calibration_datav3.txt`. Baked
//!   into the binary via `include_str!` so the binary is self-contained.
//!   SHA-256 of the baked content:
//!   `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`
//!   (~273 KB of mixed-domain prose; default for I-tier APEX).
//!
//! * [`CorpusSource::Mudler`] — mudler/apex-quant's published calibration
//!   set. Per ADR-033 §Pi, this is "sampled from openassistant +
//!   the-stack-smol + math-instruct + ToolBench, version-pinned in
//!   `data/calibration/mudler_v1.README.md` with sampling seed +
//!   per-source token counts". In Phase A v1 the canonical mudler corpus
//!   has NOT been collected (it requires multi-source sampling beyond
//!   the scope of a single iter). The variant resolves to
//!   [`ImatrixError::UnknownBakedCorpus`] with a pointer to ADR-033 §Pi
//!   for the operator to assemble it manually.
//!
//! * [`CorpusSource::UserFile`] — operator-supplied `.txt` path. Loaded
//!   from disk at runtime.
//!
//! ## Tokenization
//!
//! Per ADR-033 §Pi the chunk size is `n_ctx / n_parallel` (default
//! `n_parallel=1` so `chunk_size == n_ctx`). Tokenization itself is
//! Phase B work (it requires the per-arch tokenizer loaded from the HF
//! directory — same code path as `src/convert/tokenizer.rs` but at
//! token-id resolution time, not metadata-emission time). Phase A
//! exposes the raw UTF-8 [`CorpusBytes`] payload; the Phase B driver
//! consumes it.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every read failure
//! surfaces as a typed [`ImatrixError`] — no silent skip.

use std::path::{Path, PathBuf};

use super::error::ImatrixError;

/// Baked `cdv3.txt` corpus (bartowski's `calibration_datav3.txt`).
///
/// Source: <https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8>
/// SHA-256 (verified at vendor time):
///   `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`
const CDV3_CORPUS: &str = include_str!("../../../data/calibration/cdv3.txt");

/// The set of baked-corpus CLI names. Used in [`ImatrixError::UnknownBakedCorpus`].
pub const BAKED_CORPUS_NAMES: &[&str] = &["cdv3", "mudler", "user-file"];

/// Operator-facing corpus source selector.
///
/// Mirrors the `--imatrix-corpus {cdv3,mudler,user-file}` CLI surface.
#[derive(Debug, Clone)]
pub enum CorpusSource {
    /// Default — bartowski's `calibration_datav3.txt` baked at build time.
    Cdv3,
    /// mudler-style sampled corpus. NOT BUNDLED in Phase A (see module
    /// doc). Asking for this variant returns an error until the corpus
    /// has been collected and added to `data/calibration/mudler.txt`.
    Mudler,
    /// Operator-supplied `.txt` file path. Read at runtime.
    UserFile(PathBuf),
}

impl CorpusSource {
    /// Parse a CLI selector string into a [`CorpusSource`].
    ///
    /// `cdv3` / `mudler` are the two baked-corpus selectors.
    /// `user-file:<path>` is the operator-supplied form. Anything else
    /// is [`ImatrixError::UnknownBakedCorpus`].
    pub fn from_cli(s: &str) -> Result<Self, ImatrixError> {
        match s {
            "cdv3" => Ok(CorpusSource::Cdv3),
            "mudler" => Ok(CorpusSource::Mudler),
            other => {
                if let Some(path) = other.strip_prefix("user-file:") {
                    return Ok(CorpusSource::UserFile(PathBuf::from(path)));
                }
                Err(ImatrixError::UnknownBakedCorpus {
                    name: s.to_string(),
                    supported: BAKED_CORPUS_NAMES,
                })
            }
        }
    }

    /// Operator-visible name for diagnostics / `imatrix.datasets`
    /// metadata. For baked corpora returns the canonical CLI name
    /// (`"cdv3"`, `"mudler"`). For [`CorpusSource::UserFile`] returns the
    /// file-stem (or the full path if the stem is unavailable).
    pub fn dataset_label(&self) -> String {
        match self {
            CorpusSource::Cdv3 => "cdv3".to_string(),
            CorpusSource::Mudler => "mudler".to_string(),
            CorpusSource::UserFile(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_else(|| p.to_str().unwrap_or("user-file"))
                .to_string(),
        }
    }
}

/// Owned UTF-8 corpus payload.
///
/// `'static` for the baked variants, `'a` for user-file variants.
/// In v1 we collapse to `String` regardless — the baked corpora are
/// small enough that the extra allocation is negligible.
#[derive(Debug, Clone)]
pub struct CorpusBytes {
    /// The UTF-8 text content of the corpus.
    pub text: String,
    /// Operator-visible label propagated into `imatrix.datasets`
    /// metadata. See [`CorpusSource::dataset_label`].
    pub label: String,
}

impl CorpusBytes {
    /// Load a corpus from its [`CorpusSource`] selector.
    ///
    /// * `Cdv3`: returns the baked text immediately, no I/O.
    /// * `Mudler`: returns [`ImatrixError::UnknownBakedCorpus`] in Phase A
    ///   (the corpus has not been collected; see module doc).
    /// * `UserFile(path)`: reads the file and returns its UTF-8 text.
    ///   Per [[feedback-no-loop-suppression-2026-05-17]] any read /
    ///   utf-8 failure surfaces as [`ImatrixError::CorpusRead`].
    pub fn load(source: &CorpusSource) -> Result<Self, ImatrixError> {
        match source {
            CorpusSource::Cdv3 => Ok(CorpusBytes {
                text: CDV3_CORPUS.to_string(),
                label: "cdv3".to_string(),
            }),
            CorpusSource::Mudler => {
                // The CLI parser accepts `mudler` as a known name (so the
                // operator doesn't see "unknown corpus") but the corpus
                // itself isn't bundled in Phase A — assembling it requires
                // multi-source sampling per ADR-033 §Pi. Surface a
                // specific error pointing at the workaround.
                Err(ImatrixError::CorpusRead {
                    path: "<baked:mudler>".to_string(),
                    detail:
                        "mudler corpus is not bundled in Phase A. \
                         Either use `cdv3` (default) or supply your own via \
                         `user-file:<path>`. See ADR-033 §Pi for the mudler-style \
                         sampling recipe."
                            .to_string(),
                })
            }
            CorpusSource::UserFile(path) => load_user_file(path),
        }
    }

    /// Returns the corpus text as a `&str`. Cheap.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Approximate token count via whitespace splitting. ONLY used for
    /// pre-run sanity messages — real tokenization is per-arch and
    /// happens in the Phase B driver against the model's tokenizer.
    pub fn approx_word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Approximate byte count.
    pub fn byte_count(&self) -> usize {
        self.text.len()
    }
}

fn load_user_file(path: &Path) -> Result<CorpusBytes, ImatrixError> {
    let text = std::fs::read_to_string(path).map_err(|e| ImatrixError::CorpusRead {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let label = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("user-file")
        .to_string();
    Ok(CorpusBytes { text, label })
}

/// Chunk a tokenized corpus into `n_token`-sized chunks per ADR-033 §Pi.
///
/// Mirrors `llama-imatrix`'s chunking: `chunk_size = n_ctx / n_parallel`
/// (with default `n_parallel=1` ⇒ `chunk_size == n_ctx`). Partial trailing
/// chunks are dropped (matches `imatrix.cpp:960`).
///
/// Phase B uses this for the per-arch tokenized corpus; Phase A exposes it
/// pre-built for unit tests + future wiring.
pub fn chunk_tokens(tokens: &[u32], chunk_size: usize) -> Vec<&[u32]> {
    if chunk_size == 0 {
        return vec![];
    }
    tokens.chunks_exact(chunk_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked `cdv3` corpus is non-empty and roughly the expected
    /// size (bartowski's gist is ~273 KB / ~50k words). Catches an
    /// accidentally-empty `data/calibration/cdv3.txt` at build time.
    #[test]
    fn cdv3_baked_corpus_is_non_trivial() {
        let bytes = CorpusBytes::load(&CorpusSource::Cdv3).unwrap();
        assert_eq!(bytes.label, "cdv3");
        assert!(
            bytes.byte_count() > 100_000,
            "baked cdv3.txt unexpectedly small: {} bytes",
            bytes.byte_count()
        );
        assert!(
            bytes.approx_word_count() > 10_000,
            "baked cdv3.txt unexpectedly few words: {}",
            bytes.approx_word_count()
        );
    }

    /// `mudler` is NOT bundled in Phase A. Confirm the typed error
    /// (a `CorpusRead` carrying the not-yet-bundled diagnostic, not a
    /// silent fallback to `cdv3`).
    #[test]
    fn mudler_corpus_not_yet_bundled() {
        let err = CorpusBytes::load(&CorpusSource::Mudler).unwrap_err();
        match err {
            ImatrixError::CorpusRead { path, detail } => {
                assert_eq!(path, "<baked:mudler>");
                assert!(detail.contains("not bundled"));
            }
            other => panic!("expected CorpusRead, got {other:?}"),
        }
    }

    /// CLI parsing accepts the baked names and the `user-file:` prefix.
    #[test]
    fn from_cli_parses_baked_and_user_file() {
        assert!(matches!(CorpusSource::from_cli("cdv3"), Ok(CorpusSource::Cdv3)));
        assert!(matches!(CorpusSource::from_cli("mudler"), Ok(CorpusSource::Mudler)));
        let uf = CorpusSource::from_cli("user-file:/tmp/x.txt").unwrap();
        match uf {
            CorpusSource::UserFile(p) => assert_eq!(p.to_str(), Some("/tmp/x.txt")),
            other => panic!("expected UserFile, got {other:?}"),
        }
        // Unknown selector → typed error.
        let err = CorpusSource::from_cli("wikitext").unwrap_err();
        assert!(matches!(err, ImatrixError::UnknownBakedCorpus { .. }));
    }

    /// `dataset_label` matches what the CLI accepts (round-trip
    /// invariant for the baked variants).
    #[test]
    fn dataset_label_round_trips_for_baked() {
        assert_eq!(CorpusSource::Cdv3.dataset_label(), "cdv3");
        assert_eq!(CorpusSource::Mudler.dataset_label(), "mudler");
        let uf = CorpusSource::UserFile(PathBuf::from("/x/y/my_corpus.txt"));
        assert_eq!(uf.dataset_label(), "my_corpus.txt");
    }

    /// Loading a missing user-file returns a typed CorpusRead error
    /// (not a panic). Per [[feedback-no-loop-suppression-2026-05-17]].
    #[test]
    fn missing_user_file_errors_typed() {
        let err = CorpusBytes::load(&CorpusSource::UserFile(PathBuf::from(
            "/nonexistent/path/should/not/be/here.txt",
        )))
        .unwrap_err();
        assert!(matches!(err, ImatrixError::CorpusRead { .. }));
    }

    /// `chunk_tokens` returns exact-size chunks only; partial trailing
    /// chunks dropped (matches llama-imatrix semantics).
    #[test]
    fn chunk_tokens_drops_partial_trailing() {
        let toks: Vec<u32> = (0u32..10).collect();
        let chunks = chunk_tokens(&toks, 3);
        // 10 / 3 = 3 full chunks (9 tokens used), last token dropped.
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], &[0u32, 1, 2][..]);
        assert_eq!(chunks[1], &[3u32, 4, 5][..]);
        assert_eq!(chunks[2], &[6u32, 7, 8][..]);
    }

    /// `chunk_tokens(_, 0)` returns empty (defensive — won't divide
    /// by zero).
    #[test]
    fn chunk_tokens_zero_chunk_size() {
        let toks = vec![1u32, 2, 3];
        assert!(chunk_tokens(&toks, 0).is_empty());
    }
}
