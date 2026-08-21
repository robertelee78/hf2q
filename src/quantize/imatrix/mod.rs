//! In-tree imatrix subsystem (ADR-033 §Pi).
//!
//! ## Status
//!
//! **Phase A: SHIPPED** — corpus loader, per-tensor accumulator,
//! `.imatrix.gguf` writer + loader. Operators can:
//!
//! * Load a pre-computed `.imatrix.gguf` via `--imatrix <path>` (e.g. one
//!   produced by stock `llama-imatrix`) and use it with I-tier APEX
//!   variants.
//! * Round-trip through [`gguf_writer::write_imatrix`] +
//!   [`gguf_loader::LoadedImatrix::load_from_path`] for testing.
//! * Construct an [`accumulator::AccumulatorRegistry`] programmatically
//!   (used by Phase B's forward-pass driver — sketched in
//!   [`forward`]).
//!
//! **Phase B: DEFERRED** — forward-pass interception into hf2q's per-arch
//! decoder (`src/inference/models/<arch>/`). See [`forward`] for the
//! contract sketch and the operator-facing workaround (run stock
//! `llama-imatrix` against a F16 GGUF, then feed the result back via
//! `--imatrix`).
//!
//! ## Crate-level public API
//!
//! ```ignore
//! use crate::quantize::imatrix::{
//!     CorpusSource, CorpusBytes,           // corpus loader
//!     Accumulator, AccumulatorRegistry,    // per-tensor accumulator
//!     write_imatrix, write_imatrix_to_path,// serialize
//!     LoadedImatrix,                       // deserialize
//!     ImatrixError,                        // typed errors
//!     ComputeImatrixParams, compute_imatrix, // Phase B driver (stub)
//! };
//! ```
//!
//! ## Per-tensor algorithm reference
//!
//! Mirrors `llama-imatrix`. The on-disk schema is identical, modulo
//! FP accumulation order (ADR-033 §Pi "Risk 2 — Metal-vs-CPU activation
//! order").

pub mod accumulator;
pub mod corpus;
pub mod error;
pub mod forward;
pub mod gguf_loader;
pub mod gguf_writer;

pub use accumulator::{Accumulator, AccumulatorRegistry};
pub use corpus::{CorpusBytes, CorpusSource, BAKED_CORPUS_NAMES};
pub use error::ImatrixError;
pub use forward::{
    compute_imatrix, intercept_qmatmul_id_with_hint, intercept_qmatmul_with_hint, is_active,
    with_collector, ComputeImatrixParams, ImatrixCollector, ImatrixHint,
};
pub use gguf_loader::LoadedImatrix;
pub use gguf_writer::{write_imatrix, write_imatrix_to_path};

/// Provenance of the per-tensor imatrix data threaded into
/// [`crate::quantize::ggml_quants::apex::ApexPolicy::new_with_imatrix`].
///
/// Tracks WHERE the imatrix data came from so error messages can
/// distinguish "operator passed `--imatrix <bad-file>`" from
/// "in-tree generation produced a bad result". The data itself lives
/// in [`LoadedImatrix`].
#[derive(Debug, Clone)]
pub enum ImatrixProvenance {
    /// Loaded from disk via `--imatrix <path>` (typically a
    /// `llama-imatrix` reference output).
    LoadedFromFile {
        path: std::path::PathBuf,
        bytes: u64,
        sha256: String,
    },
    /// Computed in-tree by Phase B's forward-pass driver. Phase A does
    /// not produce this variant.
    Computed { corpus_label: String, n_ctx: u32 },
}

impl ImatrixProvenance {
    /// Human-readable label used in diagnostic messages.
    pub fn label(&self) -> String {
        match self {
            ImatrixProvenance::LoadedFromFile {
                path,
                bytes,
                sha256,
            } => format!("file:{}[bytes={},sha256={}]", path.display(), bytes, sha256),
            ImatrixProvenance::Computed {
                corpus_label,
                n_ctx,
            } => {
                format!("computed[{}@n_ctx={}]", corpus_label, n_ctx)
            }
        }
    }
}

/// Top-level public data record handed to the convert orchestrator when
/// an I-tier APEX run has imatrix data available. Composes
/// [`LoadedImatrix`] (the per-tensor data) with [`ImatrixProvenance`]
/// (where it came from) into a single value the policy layer can ingest.
#[derive(Debug)]
pub struct ImatrixData {
    /// Per-tensor accumulators (sum-of-squares + per-mat counts).
    pub loaded: LoadedImatrix,
    /// Provenance for diagnostics.
    pub provenance: ImatrixProvenance,
}

impl ImatrixData {
    /// Load an imatrix from an immutable private snapshot and record the
    /// `--imatrix <path>` provenance. The operator path is copied exactly once;
    /// parsing never reopens it, so replacement during conversion cannot make
    /// the materialized registry disagree with the recorded digest.
    pub fn load_from_path(path: &std::path::Path) -> Result<Self, ImatrixError> {
        let snapshot = snapshot_file(path)?;
        Self::load_from_snapshot(path, &snapshot)
    }

    fn load_from_snapshot(
        operator_path: &std::path::Path,
        snapshot: &ImatrixSnapshot,
    ) -> Result<Self, ImatrixError> {
        let loaded = LoadedImatrix::load_from_path_with_source(
            snapshot.file.path(),
            operator_path.display().to_string(),
        )?;
        Ok(ImatrixData {
            loaded,
            provenance: ImatrixProvenance::LoadedFromFile {
                path: operator_path.to_path_buf(),
                bytes: snapshot.bytes,
                sha256: snapshot.sha256.clone(),
            },
        })
    }

    /// Serialize the imatrix back to a `.imatrix.gguf` (e.g. for the
    /// `--imatrix-out <path>` round-trip side-effect; or when Phase B
    /// computes the imatrix in-tree and the operator passed
    /// `--imatrix-out`).
    pub fn write_gguf(
        &self,
        path: &std::path::Path,
        datasets: &[String],
    ) -> Result<(), ImatrixError> {
        write_imatrix_to_path(
            path,
            &self.loaded.registry,
            datasets,
            self.loaded.chunk_count,
            self.loaded.chunk_size,
        )
    }

    /// Number of tensor pairs (in_sum2 + counts) in this imatrix.
    pub fn tensor_pair_count(&self) -> usize {
        self.loaded.tensor_pair_count()
    }
}

struct ImatrixSnapshot {
    file: tempfile::NamedTempFile,
    bytes: u64,
    sha256: String,
}

fn snapshot_file(path: &std::path::Path) -> Result<ImatrixSnapshot, ImatrixError> {
    use std::io::{Read, Write};

    use sha2::{Digest, Sha256};

    let mut source = std::fs::File::open(path)?;
    let mut snapshot = tempfile::Builder::new()
        .prefix("hf2q-imatrix-snapshot-")
        .tempfile()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        snapshot.write_all(&buffer[..read])?;
        hasher.update(&buffer[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| ImatrixError::Parse {
                detail: format!("{} is too large to snapshot", path.display()),
            })?;
    }
    snapshot.flush()?;
    snapshot.as_file().sync_all()?;
    Ok(ImatrixSnapshot {
        file: snapshot,
        bytes,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: build an in-memory registry → write to disk →
    /// reload via [`ImatrixData::load_from_path`] → re-serialize via
    /// [`ImatrixData::write_gguf`] → reload again. Asserts the
    /// double-round-trip is byte-stable on the on-disk f32 payloads
    /// (sanity for the byte-cmp gate against `llama-imatrix`).
    #[test]
    fn imatrix_data_round_trip_is_byte_stable() {
        let mut reg = AccumulatorRegistry::new();
        let acc = reg.register("blk.0.attn_q.weight", 4, 1).unwrap();
        acc.absorb_dense(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        acc.absorb_dense(&[0.5, 0.5, 0.5, 0.5]).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_imatrix_to_path(tmp.path(), &reg, &["cdv3".to_string()], 1, 512).unwrap();

        let data = ImatrixData::load_from_path(tmp.path()).unwrap();
        assert_eq!(data.tensor_pair_count(), 1);
        let expected = snapshot_file(tmp.path()).unwrap();
        match &data.provenance {
            ImatrixProvenance::LoadedFromFile { bytes, sha256, .. } => {
                assert_eq!(*bytes, expected.bytes);
                assert_eq!(*sha256, expected.sha256);
            }
            other => panic!("expected loaded-file provenance, got {other:?}"),
        }

        let tmp2 = tempfile::NamedTempFile::new().unwrap();
        data.write_gguf(tmp2.path(), &["cdv3".to_string()]).unwrap();

        // Compare the on-disk byte streams. Note: header offsets MAY
        // differ across writes if the underlying mlx_native::gguf
        // reader normalizes anything; in practice for the closed
        // round-trip they match.
        let a = std::fs::read(tmp.path()).unwrap();
        let b = std::fs::read(tmp2.path()).unwrap();
        assert_eq!(
            a,
            b,
            "imatrix file double-round-trip should be byte-identical \
             (a.len={}, b.len={})",
            a.len(),
            b.len()
        );
    }

    #[test]
    fn immutable_snapshot_binds_the_materialized_registry() {
        let mut original_registry = AccumulatorRegistry::new();
        original_registry
            .register("blk.0.attn_q.weight", 2, 1)
            .unwrap()
            .absorb_dense(&[1.0, 2.0])
            .unwrap();
        let original = tempfile::NamedTempFile::new().unwrap();
        write_imatrix_to_path(
            original.path(),
            &original_registry,
            &["original".to_string()],
            1,
            128,
        )
        .unwrap();

        let snapshot = snapshot_file(original.path()).unwrap();

        let mut replacement_registry = AccumulatorRegistry::new();
        replacement_registry
            .register("blk.0.attn_q.weight", 2, 1)
            .unwrap()
            .absorb_dense(&[9.0, 10.0])
            .unwrap();
        write_imatrix_to_path(
            original.path(),
            &replacement_registry,
            &["replacement".to_string()],
            2,
            256,
        )
        .unwrap();

        let data = ImatrixData::load_from_snapshot(original.path(), &snapshot).unwrap();
        assert_eq!(data.loaded.datasets, ["original"]);
        assert_eq!(data.loaded.chunk_count, 1);
        assert_eq!(data.loaded.chunk_size, 128);
        assert_eq!(
            data.loaded
                .accumulator("blk.0.attn_q.weight")
                .unwrap()
                .values,
            [1.0, 4.0]
        );
        match data.provenance {
            ImatrixProvenance::LoadedFromFile {
                path,
                bytes,
                sha256,
            } => {
                assert_eq!(path, original.path());
                assert_eq!(bytes, snapshot.bytes);
                assert_eq!(sha256, snapshot.sha256);
            }
            other => panic!("expected loaded-file provenance, got {other:?}"),
        }
    }

    /// Provenance labels are stable + diagnostic-friendly.
    #[test]
    fn provenance_labels() {
        let from_file = ImatrixProvenance::LoadedFromFile {
            path: std::path::PathBuf::from("/tmp/foo.imatrix.gguf"),
            bytes: 42,
            sha256: "ab".repeat(32),
        };
        assert!(from_file.label().starts_with("file:"));
        assert!(from_file.label().contains("bytes=42"));
        assert!(from_file.label().contains(&"ab".repeat(32)));
        let computed = ImatrixProvenance::Computed {
            corpus_label: "cdv3".to_string(),
            n_ctx: 512,
        };
        assert!(computed.label().contains("cdv3"));
        assert!(computed.label().contains("512"));
    }
}
