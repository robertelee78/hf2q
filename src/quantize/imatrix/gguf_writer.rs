//! `.imatrix.gguf` writer — produces the exact schema that
//! `/opt/llama.cpp/tools/imatrix/imatrix.cpp::save_imatrix` emits.
//!
//! Per ADR-033 §Pi the on-disk schema is:
//!
//! * **KV header**:
//!   * `general.type` = `"imatrix"` (string)
//!   * `imatrix.datasets` (array of strings — calibration corpora labels)
//!   * `imatrix.chunk_count` (u32) — last_chunk count, ≈ tokens / chunk_size
//!   * `imatrix.chunk_size`  (u32) — `n_ctx / n_parallel`
//!
//! * **Per source-tensor pair**:
//!   * `<tensor_name>.in_sum2` — f32 tensor, shape `[n_per_row, n_mat]`
//!     (in PyTorch order; GGUF writes innermost-first so the on-disk
//!     dims are `[n_per_row, n_mat]` literally — see
//!     [`super::accumulator::Accumulator`]).
//!   * `<tensor_name>.counts`  — f32 tensor, shape `[1, n_mat]`
//!     (stored as float per imatrix.cpp:602's `ggml_new_tensor_2d` call).
//!
//! ## Wire-format details (cross-validated against imatrix.cpp)
//!
//! * `general.type` is written as a STRING KV pair (not part of any
//!   enum) — see imatrix.cpp:588 `gguf_set_val_str(ctx_gguf,
//!   "general.type", "imatrix");`.
//! * `imatrix.datasets` is written as a STRING ARRAY (per
//!   imatrix.cpp:590 `gguf_set_arr_str(ctx_gguf,
//!   LLM_KV_IMATRIX_DATASETS, datasets.data(), datasets.size())`).
//! * `imatrix.chunk_count` and `imatrix.chunk_size` are u32 LE.
//! * Tensor name order on disk is `std::sort` order (imatrix.cpp:568) —
//!   our [`AccumulatorRegistry`] iterates in BTreeMap (sorted) order.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: writer failures are
//! typed; never silent fallback.
//!
//! ## Phase A note
//!
//! Phase A v1 writes a STRICT-SCHEMA `.imatrix.gguf` that's byte-cmpable
//! against `llama-imatrix --output-format gguf -o ...` modulo FP
//! accumulation order. Byte-cmp gating is documented in ADR-033 §Pi
//! "Acceptance gate"; the gate is manual / CI-tagged in v1 (no
//! `llama-imatrix` binary lives in this repo's checked-in build tree
//! by default, so the gate runs against an externally-built reference).

use std::io::{Seek, Write};

use crate::backends::gguf::types::MetaValue;
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType;

use super::accumulator::{Accumulator, AccumulatorRegistry};
use super::error::ImatrixError;

/// Canonical KV keys per `tools/imatrix/imatrix.cpp:37-39`.
pub const KV_KEY_TYPE: &str = "general.type";
pub const KV_KEY_DATASETS: &str = "imatrix.datasets";
pub const KV_KEY_CHUNK_COUNT: &str = "imatrix.chunk_count";
pub const KV_KEY_CHUNK_SIZE: &str = "imatrix.chunk_size";
pub const KV_VALUE_TYPE: &str = "imatrix";

/// Render an [`AccumulatorRegistry`] into a `.imatrix.gguf` on disk.
///
/// `datasets` is the list of corpus labels propagated into
/// `imatrix.datasets` (typically a single-element vec containing the
/// `CorpusSource::dataset_label`).
///
/// `chunk_count` is the cumulative number of `chunk_size`-token chunks
/// processed across the corpus run; `chunk_size` is `n_ctx / n_parallel`
/// per ADR-033 §Pi. (At Phase A these are typically `1` and `n_ctx`
/// respectively, threaded from the driver.)
///
/// The writer wraps any `Write + Seek` sink. Use `std::fs::File` for
/// real I/O; `Cursor<Vec<u8>>` for unit tests + byte-cmp.
pub fn write_imatrix<W: Write + Seek>(
    sink: W,
    registry: &AccumulatorRegistry,
    datasets: &[String],
    chunk_count: u32,
    chunk_size: u32,
) -> Result<W, ImatrixError> {
    let mut w = GgufWriter::new(sink);

    // ---- KV section ----------------------------------------------------
    //
    // 4 KV pairs: general.type, imatrix.datasets, imatrix.chunk_count,
    // imatrix.chunk_size. Plus 2 tensor-info entries per source tensor
    // (the in_sum2 + counts pair).
    //
    // imatrix.cpp filters out zero-count tensors (line 543-565). We
    // mirror that: any accumulator with `!has_data()` is skipped so
    // the file stays consistent with llama-imatrix output (which writes
    // entries even with partial data, but `save_imatrix` at line 596
    // iterates `to_store` only). For Phase A we follow the more
    // conservative rule of skipping completely-empty tensors so that
    // a partial Phase B run produces a valid file.
    let storable: Vec<(&str, &Accumulator)> = registry
        .iter()
        .filter(|(_, acc)| acc.has_data())
        .collect();
    let tensor_count = storable.len() as u64 * 2; // each pair contributes 2 tensors
    let kv_count: u64 = 4;
    w.write_header(tensor_count, kv_count)?;

    // general.type = "imatrix"
    w.write_metadata_kv(KV_KEY_TYPE, &MetaValue::String(KV_VALUE_TYPE.to_string()))?;

    // imatrix.datasets = [<corpora>]  (array of strings)
    w.write_metadata_kv(
        KV_KEY_DATASETS,
        &MetaValue::ArrayString(datasets.to_vec()),
    )?;

    // imatrix.chunk_count = <u32>
    w.write_metadata_kv(KV_KEY_CHUNK_COUNT, &MetaValue::U32(chunk_count))?;

    // imatrix.chunk_size = <u32>
    w.write_metadata_kv(KV_KEY_CHUNK_SIZE, &MetaValue::U32(chunk_size))?;

    // ---- Tensor-info reservations -------------------------------------
    //
    // Per imatrix.cpp:601-604:
    //   in_sum2 = ggml_new_tensor_2d(F32, n_per_row, n_mat)
    //   counts  = ggml_new_tensor_2d(F32, 1, n_mat)
    //
    // GGUF stores dims innermost-first; PyTorch shape [n_per_row, n_mat]
    // → GGUF dims [n_per_row, n_mat] (already innermost-first since
    // n_per_row is the row stride).
    //
    // We record (idx_in_sum2, idx_counts) per accumulator so the streaming
    // step below knows which tensor index each payload belongs to.
    let mut payload_idx: Vec<(usize, usize)> = Vec::with_capacity(storable.len());
    for (_, acc) in &storable {
        let in_sum2_name = format!("{}.in_sum2", acc.name);
        let counts_name = format!("{}.counts", acc.name);

        // in_sum2: shape [n_per_row, n_mat], F32
        let in_sum2_idx = w.reserve_tensor_info(
            &in_sum2_name,
            &[acc.n_per_row as u64, acc.n_mat as u64],
            GgmlType::F32,
        )?;
        // counts: shape [1, n_mat], F32
        let counts_idx = w.reserve_tensor_info(
            &counts_name,
            &[1u64, acc.n_mat as u64],
            GgmlType::F32,
        )?;
        payload_idx.push((in_sum2_idx, counts_idx));
    }

    // ---- Pad to ALIGNMENT before tensor data block ---------------------
    w.pad_to_alignment()?;

    // ---- Stream tensor payloads ---------------------------------------
    //
    // imatrix.cpp:606-611:
    //   for (j = 0; j < nval; ++j) in_sum2->data[j] = stat.values[j];
    //   for (j = 0; j < nmat; ++j) counts->data[j]  = (float) stat.counts[j];
    //
    // Note counts is stored as f32, not i64. This matches llama-imatrix's
    // GGUF schema (legacy DAT format stores int32 ncalls; GGUF promotes
    // to f32 — see imatrix.cpp:602 ggml_new_tensor_2d(GGML_TYPE_F32,...)).
    for ((_, acc), (in_sum2_idx, counts_idx)) in storable.iter().zip(payload_idx.iter()) {
        // in_sum2 payload (f32 LE)
        let mut in_sum2_bytes = Vec::with_capacity(acc.values.len() * 4);
        for v in &acc.values {
            in_sum2_bytes.extend_from_slice(&v.to_le_bytes());
        }
        w.stream_tensor_payload(*in_sum2_idx, &in_sum2_bytes)?;

        // counts payload (i64 → f32 LE per imatrix.cpp:610)
        let mut counts_bytes = Vec::with_capacity(acc.counts.len() * 4);
        for &c in &acc.counts {
            counts_bytes.extend_from_slice(&(c as f32).to_le_bytes());
        }
        w.stream_tensor_payload(*counts_idx, &counts_bytes)?;
    }

    w.finalize()?;
    Ok(w.into_inner())
}

/// Convenience: write to a file path.
pub fn write_imatrix_to_path(
    path: &std::path::Path,
    registry: &AccumulatorRegistry,
    datasets: &[String],
    chunk_count: u32,
    chunk_size: u32,
) -> Result<(), ImatrixError> {
    let f = std::fs::File::create(path)?;
    write_imatrix(f, registry, datasets, chunk_count, chunk_size)?;
    Ok(())
}

/// Estimate the metadata-KV section size in bytes (used by Phase B's
/// progress prints; not load-bearing). Each KV pair encodes as
/// `u64 LE key_len | key bytes | u32 LE type | <payload>`.
pub fn estimate_kv_bytes(datasets: &[String]) -> usize {
    fn str_size(s: &str) -> usize {
        8 + s.len()
    }
    let mut bytes = 0;
    // general.type = "imatrix"
    bytes += str_size(KV_KEY_TYPE) + 4 + str_size(KV_VALUE_TYPE);
    // imatrix.datasets array-string: key + u32 type + u32 elem_type +
    // u64 elem_count + sum(str_size(elem))
    bytes += str_size(KV_KEY_DATASETS) + 4 + 4 + 8;
    for d in datasets {
        bytes += str_size(d);
    }
    // imatrix.chunk_count u32: key + u32 type + 4 bytes value
    bytes += str_size(KV_KEY_CHUNK_COUNT) + 4 + 4;
    // imatrix.chunk_size u32: key + u32 type + 4 bytes value
    bytes += str_size(KV_KEY_CHUNK_SIZE) + 4 + 4;
    bytes
}

/// Pre-validate metadata-KV encoding by writing to an in-memory buffer.
/// Used by [`gguf_writer`] tests; not load-bearing in production.
#[cfg(test)]
fn write_kv_to_vec(key: &str, value: &MetaValue) -> Vec<u8> {
    use crate::backends::gguf::types::write_metadata_kv;
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, key, value).expect("in-memory write cannot fail");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Smoke: write a tiny synthetic imatrix to memory + parse with
    /// mlx_native::gguf::GgufFile. Asserts the canonical schema is
    /// emitted (general.type, imatrix.datasets, chunk_count, chunk_size)
    /// plus the per-tensor pair (`*.in_sum2` + `*.counts`).
    #[test]
    fn round_trip_minimal_imatrix() {
        let mut reg = AccumulatorRegistry::new();
        let acc = reg.register("blk.0.attn_q.weight", 8, 1).unwrap();
        for j in 0..8 {
            // arbitrary deterministic values
            let row: Vec<f32> = (0..8).map(|k| (j + k) as f32).collect();
            acc.absorb_dense(&row).unwrap();
        }

        let buf = Cursor::new(Vec::new());
        let inner = write_imatrix(
            buf,
            &reg,
            &["cdv3".to_string()],
            /*chunk_count=*/ 1,
            /*chunk_size=*/ 512,
        )
        .unwrap();
        let bytes = inner.into_inner();

        // Persist to a temp file so the canonical reader can parse it.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            use std::io::Write;
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }

        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse imatrix gguf");
        // Metadata KV checks
        assert_eq!(gguf.metadata_string("general.type"), Some("imatrix"));
        assert_eq!(gguf.metadata_u32("imatrix.chunk_count"), Some(1));
        assert_eq!(gguf.metadata_u32("imatrix.chunk_size"), Some(512));
        // Tensor info checks — both halves of the pair must be present.
        assert_eq!(gguf.tensor_count(), 2);
        assert!(gguf.tensor_info("blk.0.attn_q.weight.in_sum2").is_some());
        assert!(gguf.tensor_info("blk.0.attn_q.weight.counts").is_some());

        // Validate the in_sum2 shape. On-disk GGUF wire dims are
        // [n_per_row, n_mat] (innermost-first); the mlx-native reader
        // reverses to outermost-first → [n_mat, n_per_row].
        // For dense (n_mat=1, n_per_row=8): on-disk [8, 1] → read as [1, 8].
        let info = gguf
            .tensor_info("blk.0.attn_q.weight.in_sum2")
            .expect("in_sum2 present");
        assert_eq!(info.shape, vec![1, 8]);
        // counts: on-disk [1, n_mat] → read as [n_mat, 1] = [1, 1] here.
        let counts_info = gguf
            .tensor_info("blk.0.attn_q.weight.counts")
            .expect("counts present");
        assert_eq!(counts_info.shape, vec![1, 1]);
    }

    /// Empty registry → file is valid GGUF with 0 tensors + the 4 KVs.
    /// Mirrors `llama-imatrix --in-file <prev> --out-file <out>` when
    /// every input was empty.
    #[test]
    fn empty_registry_writes_valid_gguf() {
        let reg = AccumulatorRegistry::new();
        let buf = Cursor::new(Vec::new());
        let inner = write_imatrix(buf, &reg, &["cdv3".to_string()], 0, 512).unwrap();
        let bytes = inner.into_inner();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            use std::io::Write;
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }
        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse empty imatrix");
        assert_eq!(gguf.metadata_string("general.type"), Some("imatrix"));
        assert_eq!(gguf.tensor_count(), 0);
    }

    /// MoE accumulator (n_mat > 1) writes the right shape on disk.
    #[test]
    fn moe_accumulator_writes_per_expert_shape() {
        let mut reg = AccumulatorRegistry::new();
        let acc = reg
            .register("blk.0.ffn_gate_exps.weight", 4, /*n_experts=*/ 8)
            .unwrap();
        // Sprinkle activations across 2 experts to ensure has_data().
        acc.absorb_moe(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        acc.absorb_moe(3, &[0.5, 0.5, 0.5, 0.5]).unwrap();
        let buf = Cursor::new(Vec::new());
        let inner = write_imatrix(buf, &reg, &["cdv3".to_string()], 1, 512).unwrap();
        let bytes = inner.into_inner();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            use std::io::Write;
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }
        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse moe imatrix");
        let info = gguf
            .tensor_info("blk.0.ffn_gate_exps.weight.in_sum2")
            .unwrap();
        // GGUF wire dims [n_per_row=4, n_mat=8]; reader reverses to
        // [n_mat, n_per_row] = [8, 4].
        assert_eq!(info.shape, vec![8, 4]);
        let counts = gguf
            .tensor_info("blk.0.ffn_gate_exps.weight.counts")
            .unwrap();
        // GGUF wire dims [1, n_mat=8]; reader reverses to [n_mat, 1] = [8, 1].
        assert_eq!(counts.shape, vec![8, 1]);
    }

    /// `estimate_kv_bytes` matches the actual byte count produced by
    /// `write_metadata_kv` for the 4 canonical KV pairs. Catches drift
    /// between the estimator and the writer.
    #[test]
    fn estimate_kv_bytes_is_accurate() {
        let datasets = vec!["cdv3".to_string(), "mudler".to_string()];
        let mut actual = 0;
        actual += write_kv_to_vec(KV_KEY_TYPE, &MetaValue::String(KV_VALUE_TYPE.to_string())).len();
        actual += write_kv_to_vec(KV_KEY_DATASETS, &MetaValue::ArrayString(datasets.clone())).len();
        actual += write_kv_to_vec(KV_KEY_CHUNK_COUNT, &MetaValue::U32(42)).len();
        actual += write_kv_to_vec(KV_KEY_CHUNK_SIZE, &MetaValue::U32(512)).len();

        let estimate = estimate_kv_bytes(&datasets);
        assert_eq!(estimate, actual, "estimate {estimate} vs actual {actual}");
    }
}
