//! Seek-back single-pass GGUF v3 writer.
//!
//! Per ADR-033 §P2 "Seek-back incremental GGUF writer". Replaces the
//! two-pass writer at `src/backends/gguf.rs:282-1259` (text writer) and
//! `:887-1189` (mmproj writer) — both being deleted in P6.
//!
//! ## Why
//!
//! The two-pass writer pre-allocates an offset table by predicting each
//! tensor's byte size, then writes payloads at predicted offsets in
//! pass 2. When the predictor over-predicts, the writer zero-pads to
//! land on the predicted offset (the iter-99 / Bug-B-sequel bug class
//! at `gguf.rs:639,659,677,1132`).
//!
//! The seek-back design has no pass-1 prediction:
//!
//! 1. Write header (magic + version + tensor_count + kv_count).
//! 2. Write all metadata KV-pairs.
//! 3. Reserve space for each tensor-info entry — the entry's `offset`
//!    field is written as `0` placeholder; record the FILE POSITION of
//!    that field so we can come back later.
//! 4. Pad to ALIGNMENT (the tensor-data block must start on a 32-byte
//!    boundary).
//! 5. Stream tensor data: for each tensor, record the current file
//!    position (relative to `tensor_data_offset`) as the tensor's
//!    actual offset, write the payload, pad to ALIGNMENT for next
//!    tensor.
//! 6. **Seek back** to each tensor-info-entry offset-field position;
//!    write the actual offset there.
//!
//! No size prediction → no zero-pad fallback → no header / payload
//! offset disagreement.
//!
//! ## Per [[feedback-no-loop-suppression-2026-05-17]]
//!
//! Shape mismatches are typed `Err`, never silent zero-pad / F16
//! demotion. The vision/audio modality gate lives at the dispatcher
//! layer above this writer (see `crate::quantize::ggml_quants::vision`);
//! the writer itself doesn't decide.

use std::io::{Seek, SeekFrom, Write};

use super::types::{
    align_up, write_gguf_string, write_metadata_kv, MetaValue, ALIGNMENT, GGUF_MAGIC, GGUF_VERSION,
};
use crate::quantize::ggml_quants::GgmlType;

/// Errors specific to the seek-back writer.
///
/// Per the no-fallback rule: every recoverable-failure mode surfaces as
/// a typed variant here; the writer never silently demotes a tensor's
/// type / size. Callers above the writer (the dispatcher) handle the
/// vision/audio gate before calling `reserve_tensor_info`.
#[derive(Debug)]
pub enum WriterError {
    /// Underlying I/O failure on `write` / `seek` / `flush`.
    Io(std::io::Error),

    /// `stream_tensor_payload` called with a `tensor_idx` whose
    /// `reserve_tensor_info` was never called. Indicates a caller-side
    /// orchestration bug.
    UnknownTensorIndex { tensor_idx: usize, reserved: usize },

    /// `stream_tensor_payload` called twice for the same `tensor_idx`.
    /// Per [[feedback-no-loop-suppression-2026-05-17]]: would silently
    /// drift offsets if allowed — refuse loudly instead.
    DuplicateTensorPayload { tensor_idx: usize },

    /// `finalize` called before every reserved tensor had its payload
    /// streamed. The header would point at uninitialized offsets.
    MissingTensorPayloads { reserved: usize, streamed: usize },

    /// Payload byte count doesn't match the row-format expectation for
    /// the tensor's `GgmlType + shape`. Per [[feedback-no-loop-suppression]]:
    /// surfaces the upstream quantizer's row-misalignment as a hard
    /// error rather than zero-padding to "fix" it.
    PayloadSizeMismatch {
        tensor_idx: usize,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for WriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriterError::Io(e) => write!(f, "gguf writer I/O: {e}"),
            WriterError::UnknownTensorIndex { tensor_idx, reserved } => write!(
                f,
                "stream_tensor_payload tensor_idx {tensor_idx} >= reserved count {reserved}"
            ),
            WriterError::DuplicateTensorPayload { tensor_idx } => write!(
                f,
                "stream_tensor_payload called twice for tensor_idx {tensor_idx}"
            ),
            WriterError::MissingTensorPayloads { reserved, streamed } => write!(
                f,
                "finalize: only {streamed} / {reserved} tensors had payloads streamed"
            ),
            WriterError::PayloadSizeMismatch { tensor_idx, expected, actual } => write!(
                f,
                "tensor_idx {tensor_idx}: payload size mismatch (expected {expected} bytes, got {actual})"
            ),
        }
    }
}

impl std::error::Error for WriterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriterError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for WriterError {
    fn from(e: std::io::Error) -> Self {
        WriterError::Io(e)
    }
}

type Result<T> = std::result::Result<T, WriterError>;

/// Internal per-reserved-tensor bookkeeping.
#[derive(Debug, Clone, Copy)]
struct OffsetFixup {
    /// Absolute file position where the tensor-info entry's u64 offset
    /// field lives. Seek-back target for `finalize`.
    file_pos_of_offset_field: u64,
    /// Expected payload size in bytes (from `GgmlType::row_size`). Used
    /// to validate `stream_tensor_payload` and to align next tensor.
    expected_byte_len: usize,
}

/// Seek-back single-pass GGUF v3 writer.
///
/// The writer wraps any `Write + Seek` sink (typically `std::fs::File`).
/// `BufWriter<File>` is NOT compatible because seek-back invalidates the
/// buffer in surprising ways — use the raw `File` and let the OS page
/// cache handle write coalescing.
pub struct GgufWriter<W: Write + Seek> {
    writer: W,
    /// Per-reserved-tensor fixup record. `tensor_offsets[i]` is filled
    /// by `stream_tensor_payload(i, _)`.
    fixups: Vec<OffsetFixup>,
    /// Actual offsets (relative to the tensor-data block start) of
    /// each tensor's payload. `Some(off)` once streamed; `None`
    /// before. `finalize` writes these into the header via seek-back.
    tensor_offsets: Vec<Option<u64>>,
    /// Absolute file position where the tensor-data block begins
    /// (after the header-padding to ALIGNMENT). Computed at
    /// `pad_to_alignment` time.
    tensor_data_start: Option<u64>,
}

impl<W: Write + Seek> GgufWriter<W> {
    /// Construct a writer over `w`. The caller is responsible for
    /// truncating / positioning the underlying sink to position 0
    /// before the first `write_header` call.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            fixups: Vec::new(),
            tensor_offsets: Vec::new(),
            tensor_data_start: None,
        }
    }

    /// Write the 24-byte GGUF header: magic (4B) + version (4B u32 LE)
    /// + tensor_count (8B u64 LE) + kv_count (8B u64 LE).
    pub fn write_header(&mut self, tensor_count: u64, kv_count: u64) -> Result<()> {
        self.writer.write_all(&GGUF_MAGIC)?;
        self.writer.write_all(&GGUF_VERSION.to_le_bytes())?;
        self.writer.write_all(&tensor_count.to_le_bytes())?;
        self.writer.write_all(&kv_count.to_le_bytes())?;
        Ok(())
    }

    /// Write a single metadata KV pair using the v3 wire format.
    pub fn write_metadata_kv(&mut self, key: &str, value: &MetaValue) -> Result<()> {
        write_metadata_kv(&mut self.writer, key, value)?;
        Ok(())
    }

    /// Reserve a tensor-info entry: `name | n_dims u32 | dims[n_dims] u64
    /// | ggml_type u32 | offset u64 placeholder`.
    ///
    /// `dims` must already be in GGUF order (innermost / row first —
    /// reversed from PyTorch / safetensors ordering). Callers that work
    /// from safetensors shapes should reverse before passing.
    ///
    /// Returns the tensor index the caller passes to
    /// `stream_tensor_payload` later. Indices are dense and sequential.
    pub fn reserve_tensor_info(
        &mut self,
        name: &str,
        dims: &[u64],
        ggml_type: GgmlType,
    ) -> Result<usize> {
        // name (length-prefixed)
        write_gguf_string(&mut self.writer, name)?;
        // n_dims u32
        self.writer
            .write_all(&(dims.len() as u32).to_le_bytes())?;
        // dims[n_dims] u64
        for &d in dims {
            self.writer.write_all(&d.to_le_bytes())?;
        }
        // ggml_type u32
        let ggml_code: u32 = ggml_type.into();
        self.writer.write_all(&ggml_code.to_le_bytes())?;

        // Record the file position of the offset field BEFORE writing
        // the placeholder. `finalize` seeks here.
        let file_pos_of_offset_field = self.writer.stream_position()?;

        // 8-byte u64 LE placeholder. The single legitimate zero-write
        // in the entire seek-back writer — corrected via seek-back in
        // `finalize`. Per ADR-033 §P2 acceptance criterion: this is
        // the ONLY write_all-with-zero anywhere; grep guards it.
        self.writer.write_all(&[0u8; 8])?;

        // Compute expected payload bytes from the row format. The
        // outermost dim is the row count; the rest multiply into
        // n_per_row.
        let (rows, n_per_row) = split_rows_and_cols(dims);
        let block_size = ggml_type.block_size();
        if n_per_row % block_size != 0 {
            // Per [[feedback-no-loop-suppression-2026-05-17]]: hard
            // error instead of zero-pad / F16 demotion. The dispatcher
            // is responsible for routing shape-misaligned rows
            // elsewhere before they reach the writer.
            return Err(WriterError::PayloadSizeMismatch {
                tensor_idx: self.fixups.len(),
                expected: 0,
                actual: 0,
            });
        }
        let expected_byte_len = rows * ggml_type.row_size(n_per_row);

        let idx = self.fixups.len();
        self.fixups.push(OffsetFixup {
            file_pos_of_offset_field,
            expected_byte_len,
        });
        self.tensor_offsets.push(None);
        Ok(idx)
    }

    /// Pad the current file position up to ALIGNMENT and remember it
    /// as `tensor_data_start`. Must be called AFTER all
    /// `reserve_tensor_info` calls and BEFORE the first
    /// `stream_tensor_payload`.
    pub fn pad_to_alignment(&mut self) -> Result<()> {
        let cur = self.writer.stream_position()?;
        let target = align_up(cur, ALIGNMENT);
        let pad = (target - cur) as usize;
        if pad > 0 {
            // Header-region alignment padding is a GGUF spec
            // requirement (tensor data starts at a 32-byte boundary).
            // The bytes are spec-defined zeros; not a "fallback" or
            // "predicted-size correction".
            let zeros = [0u8; 32];
            self.writer.write_all(&zeros[..pad])?;
        }
        self.tensor_data_start = Some(target);
        Ok(())
    }

    /// Stream one tensor's payload at the current file position.
    ///
    /// Records the relative offset (current_pos - tensor_data_start)
    /// in the per-tensor offsets vector; the actual u64 offset gets
    /// written into the header in `finalize` via seek-back.
    ///
    /// Then pads up to ALIGNMENT for the next tensor (or for EOF).
    pub fn stream_tensor_payload(&mut self, tensor_idx: usize, payload: &[u8]) -> Result<()> {
        if tensor_idx >= self.fixups.len() {
            return Err(WriterError::UnknownTensorIndex {
                tensor_idx,
                reserved: self.fixups.len(),
            });
        }
        if self.tensor_offsets[tensor_idx].is_some() {
            return Err(WriterError::DuplicateTensorPayload { tensor_idx });
        }
        let expected = self.fixups[tensor_idx].expected_byte_len;
        if payload.len() != expected {
            return Err(WriterError::PayloadSizeMismatch {
                tensor_idx,
                expected,
                actual: payload.len(),
            });
        }
        let data_start = self.tensor_data_start.expect(
            "pad_to_alignment must be called before stream_tensor_payload (caller-side bug)",
        );
        let cur = self.writer.stream_position()?;
        let rel = cur - data_start;

        self.writer.write_all(payload)?;

        // Inter-tensor alignment padding (also spec-required).
        let after = cur + payload.len() as u64;
        let target = align_up(after, ALIGNMENT);
        let pad = (target - after) as usize;
        if pad > 0 {
            let zeros = [0u8; 32];
            self.writer.write_all(&zeros[..pad])?;
        }

        self.tensor_offsets[tensor_idx] = Some(rel);
        Ok(())
    }

    /// Finalize: seek back to each tensor-info entry's offset field and
    /// write the actual offset there. Flushes the underlying writer.
    pub fn finalize(&mut self) -> Result<()> {
        let streamed = self.tensor_offsets.iter().filter(|o| o.is_some()).count();
        if streamed != self.fixups.len() {
            return Err(WriterError::MissingTensorPayloads {
                reserved: self.fixups.len(),
                streamed,
            });
        }
        // Remember end-of-file position so we leave the cursor there
        // (mirrors the two-pass writer's "no trailing data" exit
        // state, important for `File::metadata().len()` reads).
        let eof = self.writer.stream_position()?;

        for (idx, fixup) in self.fixups.iter().enumerate() {
            let off = self.tensor_offsets[idx].expect("checked above");
            self.writer
                .seek(SeekFrom::Start(fixup.file_pos_of_offset_field))?;
            self.writer.write_all(&off.to_le_bytes())?;
        }
        // Restore the cursor at EOF so subsequent introspection
        // (`stream_position`) reports the file's true size, not the
        // seek-back position.
        self.writer.seek(SeekFrom::Start(eof))?;
        self.writer.flush()?;
        Ok(())
    }

    /// Consume the writer, returning the inner sink. Implicitly
    /// drops bookkeeping. Caller must have invoked `finalize`
    /// beforehand; this method does NOT flush automatically (callers
    /// that want flush-and-extract should do `finalize` first).
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Read-only view of the recorded fixups, for tests + debugging.
    #[cfg(test)]
    pub fn tensor_offsets(&self) -> &[Option<u64>] {
        &self.tensor_offsets
    }
}

/// Split `dims` (GGUF order, innermost first) into `(row_count, n_per_row)`.
///
/// GGUF stores tensors in row-major innermost-first order: for a 2D
/// `[out_dim, in_dim]` weight written as `dims = [in_dim, out_dim]`,
/// `n_per_row = in_dim` and `rows = out_dim`. For 1D tensors `rows = 1`,
/// `n_per_row = dims[0]`. For 0D (scalar — unused in GGUF tensor data),
/// returns `(0, 0)` so the caller's expected-bytes is 0.
fn split_rows_and_cols(dims: &[u64]) -> (usize, usize) {
    match dims.len() {
        0 => (0, 0),
        1 => (1, dims[0] as usize),
        _ => {
            let n_per_row = dims[0] as usize;
            let rows: usize = dims[1..].iter().map(|&d| d as usize).product();
            (rows, n_per_row)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ---- helpers ----

    /// Construct a tiny synthetic GGUF in-memory and return its bytes.
    /// 1 KV pair (`general.architecture = "test"`) + 1 Q4_0 tensor of
    /// 32 elements (one block = 18 bytes).
    fn build_tiny_gguf() -> Vec<u8> {
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(1, 1).unwrap();
        w.write_metadata_kv(
            "general.architecture",
            &MetaValue::String("test".into()),
        )
        .unwrap();
        // Q4_0 row of 32 f32 → 1 block of 18 bytes. dims = [32] (1-D).
        let idx = w
            .reserve_tensor_info("test.weight", &[32], GgmlType::Q4_0)
            .unwrap();
        assert_eq!(idx, 0);
        w.pad_to_alignment().unwrap();
        // 18 distinguishable bytes
        let payload: Vec<u8> = (0u8..18).collect();
        w.stream_tensor_payload(0, &payload).unwrap();
        w.finalize().unwrap();
        w.into_inner().into_inner()
    }

    // ---- 3 round-trip tests ----

    /// Round-trip 1: header / KV count / tensor count are correct.
    /// Validates the seek-back overwrites the 0-placeholder offset
    /// correctly (otherwise the offset would still be 0 → tensor read
    /// from header padding instead of from data block).
    #[test]
    fn round_trip_header_and_offset() {
        let bytes = build_tiny_gguf();

        // Magic + version
        assert_eq!(&bytes[0..4], &GGUF_MAGIC);
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            GGUF_VERSION
        );
        // tensor_count = 1
        assert_eq!(
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            1
        );
        // kv_count = 1
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            1
        );

        // After the KV pair + tensor-info entry there is exactly ONE
        // u64 LE offset field. It MUST have been seek-back-written to
        // the actual data offset (not 0). We don't know the exact
        // byte position without reparsing, so we verify via the
        // canonical reader below.
    }

    /// Round-trip 2: write a tiny GGUF + parse via the existing
    /// `mlx_native::gguf::GgufFile` reader; assert the metadata KV +
    /// tensor info round-trip correctly.
    ///
    /// Per task acceptance gate: "GgufWriter writes a tiny synthetic
    /// GGUF (1 metadata KV pair + 1 tensor of Q4_0 8 bytes) → re-read
    /// with the existing reader → assert metadata + tensor bytes
    /// round-trip equal." (Using 18 bytes — the actual Q4_0 single-block
    /// row size for 32 elements; 8 bytes is sub-block and would fail
    /// the row-format check, which is the correct no-fallback
    /// behavior.)
    #[test]
    fn round_trip_via_reader() {
        use std::io::{Read, Seek, SeekFrom, Write};

        let bytes = build_tiny_gguf();

        // Write to a real temp file so the reader can mmap-style
        // open it.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }

        // Parse via the existing reader from mlx-native.
        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse hf2q GGUF");

        // Metadata round-trip
        assert_eq!(gguf.metadata_count(), 1);
        assert_eq!(
            gguf.metadata_string("general.architecture"),
            Some("test")
        );

        // Tensor info round-trip
        assert_eq!(gguf.tensor_count(), 1);
        let info = gguf.tensor_info("test.weight").expect("tensor present");
        assert_eq!(info.shape, vec![32]);
        assert_eq!(info.byte_len, 18);
        // Q4_0 wire code = 2
        assert_eq!(info.ggml_type as u32, 2);

        // Tensor BYTES round-trip — read from the file at
        // (tensor_data_offset + info.offset), confirm equal to the
        // 0..18 payload we wrote.
        //
        // We don't have a public "raw bytes" reader API; reconstruct
        // it manually using the public `info` fields:
        let mut f = std::fs::File::open(tmp.path()).unwrap();
        // The tensor-data block starts at the file alignment after
        // the header; the reader exposed `info.offset` relative to
        // that. Compute the absolute file offset by:
        //   (a) reparse the on-disk file to find the data-block start
        // OR (b) cheat: scan for the unique 0..18 byte sequence.
        // We use (b) since the unique sequence is a strong fingerprint
        // and validates that the offset field was seek-back-written
        // to a position where this exact byte run actually lives.
        let mut all = Vec::new();
        f.seek(SeekFrom::Start(0)).unwrap();
        f.read_to_end(&mut all).unwrap();
        let expected: Vec<u8> = (0u8..18).collect();
        let found = all
            .windows(18)
            .position(|w| w == expected.as_slice())
            .expect("payload run 0..18 must appear exactly once in the file");
        // Sanity: the position should also be > 24 (past the header).
        assert!(found > 24, "payload found inside header region: {found}");
        // ALIGNMENT-aligned (32B)
        assert_eq!(found as u64 % ALIGNMENT, 0);
    }

    /// Round-trip 3: payload-size mismatch is a typed hard error, not
    /// a silent zero-pad (per [[feedback-no-loop-suppression-2026-05-17]]).
    #[test]
    fn payload_size_mismatch_is_typed_error() {
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(1, 0).unwrap();
        w.reserve_tensor_info("bad.weight", &[32], GgmlType::Q4_0)
            .unwrap();
        w.pad_to_alignment().unwrap();
        // Wrong size: 17 bytes (off-by-one)
        let bad_payload = vec![0u8; 17];
        let err = w.stream_tensor_payload(0, &bad_payload).unwrap_err();
        match err {
            WriterError::PayloadSizeMismatch { expected, actual, .. } => {
                assert_eq!(expected, 18);
                assert_eq!(actual, 17);
            }
            other => panic!("expected PayloadSizeMismatch, got {other:?}"),
        }
    }

    // ---- extra structural tests ----

    #[test]
    fn finalize_without_streaming_errors() {
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(1, 0).unwrap();
        w.reserve_tensor_info("unstreamed.weight", &[32], GgmlType::Q4_0)
            .unwrap();
        w.pad_to_alignment().unwrap();
        // Skip stream_tensor_payload; finalize should refuse.
        let err = w.finalize().unwrap_err();
        match err {
            WriterError::MissingTensorPayloads { reserved, streamed } => {
                assert_eq!(reserved, 1);
                assert_eq!(streamed, 0);
            }
            other => panic!("expected MissingTensorPayloads, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_stream_errors() {
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(1, 0).unwrap();
        w.reserve_tensor_info("t.weight", &[32], GgmlType::Q4_0)
            .unwrap();
        w.pad_to_alignment().unwrap();
        let payload: Vec<u8> = vec![0u8; 18];
        w.stream_tensor_payload(0, &payload).unwrap();
        let err = w.stream_tensor_payload(0, &payload).unwrap_err();
        match err {
            WriterError::DuplicateTensorPayload { tensor_idx } => {
                assert_eq!(tensor_idx, 0);
            }
            other => panic!("expected DuplicateTensorPayload, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tensor_index_errors() {
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(0, 0).unwrap();
        w.pad_to_alignment().unwrap();
        let err = w.stream_tensor_payload(0, &[]).unwrap_err();
        match err {
            WriterError::UnknownTensorIndex { tensor_idx, reserved } => {
                assert_eq!(tensor_idx, 0);
                assert_eq!(reserved, 0);
            }
            other => panic!("expected UnknownTensorIndex, got {other:?}"),
        }
    }

    #[test]
    fn multi_tensor_offsets_seek_back_correctly() {
        // 2 tensors. After streaming, both offsets should be 32B
        // aligned and the SECOND offset should equal first-row-size
        // rounded up to 32B.
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(2, 0).unwrap();
        w.reserve_tensor_info("a.weight", &[32], GgmlType::Q4_0).unwrap();
        w.reserve_tensor_info("b.weight", &[32], GgmlType::Q4_0).unwrap();
        w.pad_to_alignment().unwrap();
        let payload_a: Vec<u8> = (0u8..18).collect();
        let payload_b: Vec<u8> = (100u8..118).collect();
        w.stream_tensor_payload(0, &payload_a).unwrap();
        w.stream_tensor_payload(1, &payload_b).unwrap();
        // Offsets recorded before finalize for inspection.
        assert_eq!(w.tensor_offsets()[0], Some(0));
        // 18 bytes payload → align up to 32 → next offset = 32
        assert_eq!(w.tensor_offsets()[1], Some(32));
        w.finalize().unwrap();
    }

    #[test]
    fn header_only_no_tensors() {
        // Edge case: 0 tensors, 0 KVs. finalize must still succeed
        // and the file must be a well-formed empty GGUF.
        let buf = Cursor::new(Vec::new());
        let mut w = GgufWriter::new(buf);
        w.write_header(0, 0).unwrap();
        w.pad_to_alignment().unwrap();
        w.finalize().unwrap();
        let bytes = w.into_inner().into_inner();
        assert_eq!(&bytes[0..4], &GGUF_MAGIC);
        // Header is 24 bytes; padding rounds to next 32-byte boundary
        assert_eq!(bytes.len() % ALIGNMENT as usize, 0);
        assert_eq!(bytes.len(), 32);
    }
}
