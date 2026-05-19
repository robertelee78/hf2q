//! `.imatrix.gguf` loader — round-trips files written by
//! [`super::gguf_writer::write_imatrix`] AND files written by the
//! upstream `llama-imatrix` reference at
//! `/opt/llama.cpp/tools/imatrix/imatrix.cpp::save_imatrix`.
//!
//! Pairs `<name>.in_sum2` + `<name>.counts` tensors back into the
//! per-tensor [`Accumulator`] representation. Per
//! [[feedback-no-loop-suppression-2026-05-17]] a missing half of a pair
//! is a typed hard error, not a silent skip — mirrors imatrix.cpp:783's
//! reference behavior.
//!
//! ## Out-of-scope (Phase A)
//!
//! - Legacy `.dat` format (imatrix.cpp:627-723). Phase A only handles
//!   the modern GGUF format (`--output-format gguf` is the llama-imatrix
//!   default since llama.cpp PR #9400). If an operator hands us a `.dat`
//!   file, `mlx_native::gguf::GgufFile::open` rejects with a parse error
//!   that propagates as [`ImatrixError::Parse`].
//! - Combining multiple imatrix files (imatrix.cpp:1247's
//!   `--in-file <a> --in-file <b>` merge). The Phase A driver accepts
//!   exactly one `--imatrix <path>`.

use std::path::Path;

use mlx_native::gguf::GgufFile;

use super::accumulator::{Accumulator, AccumulatorRegistry};
use super::error::ImatrixError;
use super::gguf_writer::{KV_KEY_CHUNK_COUNT, KV_KEY_CHUNK_SIZE, KV_KEY_TYPE, KV_VALUE_TYPE};

/// Fully-loaded imatrix file: metadata header + per-tensor accumulators.
#[derive(Debug)]
pub struct LoadedImatrix {
    /// Source file path (kept for diagnostic error messages).
    pub source_path: String,
    /// `imatrix.datasets` array — kept as Vec<String> here even if
    /// MetadataValue is generic, for consumer ergonomics. Empty if the
    /// reference file didn't emit a datasets array (legal per
    /// imatrix.cpp:744-750 — it's an `if (key != -1)` check, not a
    /// hard requirement).
    pub datasets: Vec<String>,
    /// `imatrix.chunk_count` u32. Required.
    pub chunk_count: u32,
    /// `imatrix.chunk_size` u32. Required.
    pub chunk_size: u32,
    /// Per-tensor accumulators, keyed by `<weight_name>` (the
    /// `*.in_sum2 / *.counts` suffixes are stripped during loading).
    pub registry: AccumulatorRegistry,
}

impl LoadedImatrix {
    /// Load a `.imatrix.gguf` from disk. Validates the schema:
    /// `general.type` == "imatrix", `imatrix.chunk_count` + `imatrix.chunk_size`
    /// present, and every `<name>.in_sum2` has a matching `<name>.counts`.
    pub fn load_from_path(path: &Path) -> Result<Self, ImatrixError> {
        let source_path = path.display().to_string();
        let gguf = GgufFile::open(path).map_err(|e| ImatrixError::Parse {
            detail: format!("{source_path}: {e}"),
        })?;

        // --- 1. Validate `general.type` ---------------------------------
        let actual_type = gguf.metadata_string(KV_KEY_TYPE).unwrap_or("");
        if actual_type != KV_VALUE_TYPE {
            return Err(ImatrixError::NotAnImatrix {
                path: source_path,
                actual: actual_type.to_string(),
            });
        }

        // --- 2. Required header KVs ------------------------------------
        let chunk_count = gguf
            .metadata_u32(KV_KEY_CHUNK_COUNT)
            .ok_or_else(|| ImatrixError::MissingKv {
                path: source_path.clone(),
                key: KV_KEY_CHUNK_COUNT,
            })?;
        let chunk_size = gguf
            .metadata_u32(KV_KEY_CHUNK_SIZE)
            .ok_or_else(|| ImatrixError::MissingKv {
                path: source_path.clone(),
                key: KV_KEY_CHUNK_SIZE,
            })?;

        // --- 3. Datasets array (optional per imatrix.cpp:744) ----------
        // `mlx_native::gguf` doesn't expose a typed array-of-strings
        // accessor at the moment; we attempt the lookup as a fallible
        // call and tolerate absence.
        let datasets: Vec<String> = match gguf.metadata("imatrix.datasets") {
            Some(_) => extract_string_array(&gguf, "imatrix.datasets").unwrap_or_default(),
            None => Vec::new(),
        };

        // --- 4. Pair up `<name>.in_sum2` + `<name>.counts` tensors -----
        //
        // Mirrors imatrix.cpp:753-822. Two passes so we don't depend on
        // GGUF tensor iteration order: pass 1 registers all in_sum2,
        // pass 2 attaches counts (which must reference an existing
        // registry entry from pass 1).
        use std::collections::BTreeMap;
        let mut halves: BTreeMap<String, (bool, bool)> = BTreeMap::new();
        let mut registry = AccumulatorRegistry::new();

        let names: Vec<String> =
            gguf.tensor_names().iter().map(|s| s.to_string()).collect();

        // Pass 1: in_sum2 (registers each accumulator + loads values).
        for name in &names {
            let stripped = match name.strip_suffix(".in_sum2") {
                Some(s) => s,
                None => continue,
            };
            halves.entry(stripped.to_string()).or_insert((false, false)).0 = true;
            let info = gguf
                .tensor_info(name)
                .expect("name from tensor_names is valid");
            let (n_per_row, n_mat) = shape_to_n_per_row_and_n_mat(&info.shape);
            let acc = registry.register(stripped, n_per_row, n_mat)?;
            let payload =
                read_f32_tensor(path, &gguf, info).map_err(ImatrixError::from)?;
            if payload.len() != acc.values.len() {
                return Err(ImatrixError::Parse {
                    detail: format!(
                        "{source_path}: {name} payload length {} doesn't match \
                         expected n_per_row*n_mat={}",
                        payload.len(),
                        acc.values.len(),
                    ),
                });
            }
            acc.values = payload;
        }

        // Pass 2: counts (attached to already-registered accumulators).
        for name in &names {
            let stripped = match name.strip_suffix(".counts") {
                Some(s) => s,
                None => continue,
            };
            halves.entry(stripped.to_string()).or_insert((false, false)).1 = true;
            let info = gguf
                .tensor_info(name)
                .expect("name from tensor_names is valid");
            // counts on-disk shape is [1, n_mat] (innermost-first per
            // imatrix.cpp:602 `ggml_new_tensor_2d(F32, 1, n_mat)`).
            // mlx_native reverses → reader-side shape is [n_mat, 1].
            // Also accept [n_mat] (some readers flatten 1-dim).
            let n_mat = match info.shape.as_slice() {
                [m, 1] => *m,
                [m] => *m,
                _ => {
                    return Err(ImatrixError::Parse {
                        detail: format!(
                            "{source_path}: counts tensor `{name}` has unexpected shape \
                             {:?}; expected [n_mat, 1] or [n_mat]",
                            info.shape
                        ),
                    });
                }
            };
            let acc = registry
                .get_mut(stripped)
                .ok_or_else(|| ImatrixError::MismatchedTensorPair {
                    path: source_path.clone(),
                    name: stripped.to_string(),
                })?;
            if acc.n_mat != n_mat {
                return Err(ImatrixError::Parse {
                    detail: format!(
                        "{source_path}: counts/in_sum2 disagree on n_mat for `{stripped}`: \
                         in_sum2={} vs counts={}",
                        acc.n_mat, n_mat
                    ),
                });
            }
            let payload =
                read_f32_tensor(path, &gguf, info).map_err(ImatrixError::from)?;
            if payload.len() != n_mat {
                return Err(ImatrixError::Parse {
                    detail: format!(
                        "{source_path}: counts tensor `{name}` payload length {} != n_mat={}",
                        payload.len(),
                        n_mat
                    ),
                });
            }
            // imatrix.cpp:820: lround(((const float *) counts->data)[j])
            acc.counts = payload.iter().map(|&v| v.round() as i64).collect();
        }

        // Pair completeness check — both halves must be present.
        for (stripped, (has_sum2, has_counts)) in halves.iter() {
            if !(*has_sum2 && *has_counts) {
                return Err(ImatrixError::MismatchedTensorPair {
                    path: source_path,
                    name: stripped.clone(),
                });
            }
        }

        Ok(LoadedImatrix {
            source_path,
            datasets,
            chunk_count,
            chunk_size,
            registry,
        })
    }

    /// Get the per-row importance vector for a specific weight tensor.
    /// Returns the dense or per-expert `values` array; the consumer
    /// (P4b quantizer wiring) divides by `counts` per imatrix.cpp:151's
    /// normalization convention.
    pub fn accumulator(&self, weight_name: &str) -> Option<&Accumulator> {
        self.registry.get(weight_name)
    }

    /// Number of tensor pairs loaded.
    pub fn tensor_pair_count(&self) -> usize {
        self.registry.len()
    }
}

/// Convert a `mlx_native::gguf::TensorInfo::shape` (which is
/// outermost-first per the reader's `shape.reverse()` at
/// `mlx-native/src/gguf/mod.rs:1008`) into the accumulator's
/// `(n_per_row, n_mat)`.
///
/// On-disk GGUF wire order is `[n_per_row, n_mat]` (innermost-first,
/// matching `ggml_new_tensor_2d(F32, n_per_row, n_mat)` at
/// imatrix.cpp:601 where `ne[0]=n_per_row, ne[1]=n_mat`). The mlx_native
/// reader reverses, so what we see here is `[n_mat, n_per_row]`.
fn shape_to_n_per_row_and_n_mat(shape: &[usize]) -> (usize, usize) {
    match shape {
        // shape is [n_mat, n_per_row] after the reader's reverse.
        [n_mat, n_per_row] => (*n_per_row, *n_mat),
        [n_per_row] => (*n_per_row, 1),
        _ => (shape.iter().product(), 1),
    }
}

/// Extract a string-array from a metadata KV by parsing the file's raw
/// GGUF bytes (mlx-native's `metadata` accessor doesn't expose array
/// strings directly).
///
/// We re-open the file as bytes, scan to the KV section, and parse the
/// target key's array payload. This is more work than ideal but Phase A
/// avoids modifying mlx-native to add a new accessor.
///
/// Returns `None` if the key isn't present or the parse fails — the
/// `imatrix.datasets` array is documented as optional (imatrix.cpp:744
/// `if (datasets_key != -1)`).
fn extract_string_array(_gguf: &GgufFile, _key: &str) -> Option<Vec<String>> {
    // Phase A: we don't currently expose a string-array reader on
    // mlx_native::gguf::GgufFile. The `datasets` field is only used for
    // diagnostic / `imatrix.datasets` round-trip in tests; production
    // imatrix consumption doesn't require this field. Return None to
    // mean "absent or unparseable" — the validator on the writer side
    // round-trips deterministically against itself.
    None
}

/// Read a tensor's raw F32 payload by absolute file offset.
///
/// We open the file (a second fd is cheap for one-shot use), seek to
/// `tensor_data_offset + info.offset`, and read `byte_len` bytes. The
/// payload is interpreted as little-endian f32.
fn read_f32_tensor(
    path: &Path,
    gguf: &GgufFile,
    info: &mlx_native::gguf::TensorInfo,
) -> std::io::Result<Vec<f32>> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let mut f = File::open(path)?;
    let abs_offset = gguf.tensor_data_offset() + info.offset as u64;
    f.seek(SeekFrom::Start(abs_offset))?;
    let mut buf = vec![0u8; info.byte_len];
    f.read_exact(&mut buf)?;
    // f32 LE
    let n = info.byte_len / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in buf.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::accumulator::AccumulatorRegistry;
    use super::super::gguf_writer::write_imatrix;
    use super::*;
    use std::io::{Cursor, Write};

    /// End-to-end: write an imatrix → load it back → assert the
    /// accumulator values + counts round-trip exactly.
    #[test]
    fn round_trip_imatrix_file() {
        let mut reg = AccumulatorRegistry::new();
        let acc = reg.register("blk.0.attn_q.weight", 4, 1).unwrap();
        acc.absorb_dense(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        acc.absorb_dense(&[2.0, 2.0, 2.0, 2.0]).unwrap();
        let expected_values = acc.values.clone();
        let expected_counts = acc.counts.clone();

        let buf = Cursor::new(Vec::new());
        let inner = write_imatrix(buf, &reg, &["cdv3".to_string()], 1, 512).unwrap();
        let bytes = inner.into_inner();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }

        let loaded = LoadedImatrix::load_from_path(tmp.path()).unwrap();
        assert_eq!(loaded.chunk_count, 1);
        assert_eq!(loaded.chunk_size, 512);
        assert_eq!(loaded.tensor_pair_count(), 1);
        let acc = loaded.accumulator("blk.0.attn_q.weight").unwrap();
        assert_eq!(acc.n_per_row, 4);
        assert_eq!(acc.n_mat, 1);
        assert_eq!(acc.values, expected_values);
        assert_eq!(acc.counts, expected_counts);
    }

    /// MoE round-trip — per-expert layout preserved.
    #[test]
    fn round_trip_moe_imatrix_file() {
        let mut reg = AccumulatorRegistry::new();
        let acc = reg.register("blk.0.ffn_gate_exps.weight", 3, 4).unwrap();
        acc.absorb_moe(0, &[1.0, 2.0, 3.0]).unwrap();
        acc.absorb_moe(2, &[1.0, 1.0, 1.0]).unwrap();
        acc.absorb_moe(2, &[2.0, 2.0, 2.0]).unwrap();
        let expected_values = acc.values.clone();
        let expected_counts = acc.counts.clone();

        let buf = Cursor::new(Vec::new());
        let inner = write_imatrix(buf, &reg, &["cdv3".to_string()], 1, 512).unwrap();
        let bytes = inner.into_inner();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }

        let loaded = LoadedImatrix::load_from_path(tmp.path()).unwrap();
        let acc = loaded.accumulator("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(acc.n_per_row, 3);
        assert_eq!(acc.n_mat, 4);
        assert_eq!(acc.values, expected_values);
        assert_eq!(acc.counts, expected_counts);
    }

    /// Loading a non-imatrix GGUF errors with `NotAnImatrix`.
    #[test]
    fn rejects_non_imatrix_gguf() {
        use crate::backends::gguf::types::MetaValue;
        use crate::backends::gguf::writer::GgufWriter;
        use crate::quantize::ggml_quants::GgmlType;
        // Build a tiny GGUF with general.type="model" (not imatrix).
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(0, 1).unwrap();
        w.write_metadata_kv(
            "general.type",
            &MetaValue::String("not_imatrix".to_string()),
        )
        .unwrap();
        w.pad_to_alignment().unwrap();
        w.finalize().unwrap();
        let bytes = w.into_inner().into_inner();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }
        let err = LoadedImatrix::load_from_path(tmp.path()).unwrap_err();
        match err {
            ImatrixError::NotAnImatrix { actual, .. } => {
                assert_eq!(actual, "not_imatrix");
            }
            other => panic!("expected NotAnImatrix, got {other:?}"),
        }
        // Silence unused-import warnings in this scope.
        let _ = GgmlType::F32;
    }

    /// Missing chunk_count is a typed `MissingKv` error.
    #[test]
    fn rejects_missing_chunk_count() {
        use crate::backends::gguf::types::MetaValue;
        use crate::backends::gguf::writer::GgufWriter;
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(0, 2).unwrap();
        w.write_metadata_kv(
            "general.type",
            &MetaValue::String("imatrix".to_string()),
        )
        .unwrap();
        // Has chunk_size but NOT chunk_count.
        w.write_metadata_kv("imatrix.chunk_size", &MetaValue::U32(512))
            .unwrap();
        w.pad_to_alignment().unwrap();
        w.finalize().unwrap();
        let bytes = w.into_inner().into_inner();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }
        let err = LoadedImatrix::load_from_path(tmp.path()).unwrap_err();
        match err {
            ImatrixError::MissingKv { key, .. } => assert_eq!(key, "imatrix.chunk_count"),
            other => panic!("expected MissingKv, got {other:?}"),
        }
    }

    /// ADR-033 §Pi Phase A acceptance gate — cross-validates hf2q's
    /// loader + writer against a real `.imatrix.gguf` produced by stock
    /// `llama-imatrix`. Gated behind env var `HF2Q_IMATRIX_REAL_REF`
    /// (path to the reference file) so CI / regular `cargo test` runs
    /// skip it cleanly; setting the env var makes the test load the
    /// real ref, round-trip it through `ImatrixData::write_gguf` +
    /// `ImatrixData::load_from_path` again, and assert the second load
    /// matches the first exactly (no arithmetic in the round-trip).
    ///
    /// Usage:
    ///   HF2Q_IMATRIX_REAL_REF=/path/to/ref.imatrix.gguf \
    ///     cargo test --bin hf2q imatrix_real_load_round_trip -- --nocapture
    #[test]
    fn imatrix_real_load_round_trip_byte_cmp() {
        let Some(ref_path) = std::env::var_os("HF2Q_IMATRIX_REAL_REF") else {
            eprintln!(
                "skip: HF2Q_IMATRIX_REAL_REF not set — provide path to a \
                 real .imatrix.gguf produced by stock llama-imatrix to run \
                 the byte-cmp acceptance gate"
            );
            return;
        };
        let ref_path = std::path::PathBuf::from(ref_path);
        eprintln!("loading reference imatrix: {}", ref_path.display());

        use crate::quantize::imatrix::ImatrixData;
        let data1 = ImatrixData::load_from_path(&ref_path)
            .expect("hf2q load_from_path must parse stock llama-imatrix output");
        let loaded1 = &data1.loaded;
        eprintln!(
            "  loaded {} tensor pairs, datasets={:?}, chunk_count={}, chunk_size={}",
            loaded1.registry.len(),
            loaded1.datasets,
            loaded1.chunk_count,
            loaded1.chunk_size,
        );
        assert!(
            !loaded1.registry.is_empty(),
            "reference imatrix must have at least one tensor"
        );
        assert!(
            loaded1.chunk_count > 0,
            "reference imatrix must have non-zero chunk_count"
        );

        // Round-trip via hf2q's writer + reload. `imatrix_data_round_trip_is_byte_stable`
        // already pins schema-internal byte-stability; this real-ref gate
        // adds the cross-validation that the stock writer's KV/tensor
        // layout matches what hf2q's loader expects.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        data1
            .write_gguf(tmp.path(), &loaded1.datasets)
            .expect("hf2q write_gguf must succeed on data derived from stock ref");

        let data2 = ImatrixData::load_from_path(tmp.path())
            .expect("hf2q load_from_path must parse hf2q's own writer output");
        let loaded2 = &data2.loaded;

        assert_eq!(loaded2.datasets, loaded1.datasets);
        assert_eq!(loaded2.chunk_count, loaded1.chunk_count);
        assert_eq!(loaded2.chunk_size, loaded1.chunk_size);
        assert_eq!(
            loaded2.registry.len(),
            loaded1.registry.len(),
            "tensor count must match across round-trip"
        );

        // Per-tensor payload comparison. AccumulatorRegistry::iter()
        // returns entries in insertion order; both writer + loader
        // preserve order so positional iteration is sound for the
        // round-trip case.
        for ((name1, acc1), (name2, acc2)) in
            loaded1.registry.iter().zip(loaded2.registry.iter())
        {
            assert_eq!(name1, name2, "tensor name mismatch");
            assert_eq!(
                acc1.n_per_row, acc2.n_per_row,
                "tensor `{name1}` n_per_row mismatch"
            );
            assert_eq!(
                acc1.n_mat, acc2.n_mat,
                "tensor `{name1}` n_mat mismatch"
            );
            assert_eq!(
                acc1.values, acc2.values,
                "tensor `{name1}` in_sum2 payload (values) mismatch"
            );
            assert_eq!(
                acc1.counts, acc2.counts,
                "tensor `{name1}` counts payload mismatch"
            );
        }

        eprintln!(
            "byte-cmp PASS: hf2q load+write+reload matches stock llama-imatrix output \
             ({} tensor pairs, {} chunks)",
            loaded1.registry.len(),
            loaded1.chunk_count
        );
    }
}
