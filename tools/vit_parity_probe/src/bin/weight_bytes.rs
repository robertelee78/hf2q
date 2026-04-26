//! F16 weight byte-identity probe for ADR-005 Phase 2c iter 131 (W62).
//!
//! Question being answered: does hf2q's loader produce a byte-identical
//! `MlxBuffer` storage for an F16 weight tensor, or does the load path
//! silently transform / round / pad bytes between disk and device?
//!
//! Why we ask: W61 iter-130 enumerated four candidate explanations for
//! the residual ~17.5%/block geomean cascade after the BF16/F16 leak
//! audit. Candidate #2 was "F16 weight bytes are corrupted at load
//! time". The dtype audit at iter-130 confirmed the *types* stay F16,
//! but did NOT prove the bytes are unmodified. This probe closes that
//! gap with a one-shot byte equality check on a representative late-
//! block tensor.
//!
//! Approach (no fancy hashing — straight byte equality + first-mismatch
//! diagnostic):
//!
//!   1. Open the GGUF file via the production
//!      `mlx_native::gguf::GgufFile::open` path.
//!   2. Look up the named tensor's `TensorInfo` (offset, byte_len,
//!      shape, ggml_type).
//!   3. Read the raw bytes from disk DIRECTLY (independent path: open
//!      the file with `std::fs::File`, seek to
//!      `tensor_data_offset + info.offset`, read `info.byte_len`
//!      bytes). This bypasses the GgufFile reader entirely.
//!   4. Load the same tensor via `GgufFile::load_tensor` into an
//!      `MlxBuffer`. On Apple Silicon shared memory the
//!      `MlxBuffer::as_slice::<u8>()` view is the actual storage.
//!   5. Compare byte-for-byte. Identical → load path is byte-faithful;
//!      candidate #2 is conclusively falsified for this tensor.
//!
//! Limitation: this is a one-tensor probe. Iter-131 picks
//! `v.blk.26.attn_q.weight` because it sits at the spike boundary the
//! cascade audit is targeting. If a default-on byte-identity check is
//! ever needed across all tensors, that logic belongs in
//! `LoadedMmprojWeights::load`, not here.
//!
//! Usage:
//!   weight_bytes <gguf-path> <tensor-name>
//!
//! Exit code:
//!   0 — bytes match exactly
//!   1 — bytes differ (first-mismatch info printed)
//!   2 — I/O / lookup / API error

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::ExitCode;

use mlx_native::gguf::GgufFile;
use mlx_native::MlxDevice;

fn main() -> ExitCode {
    let argv: Vec<String> = env::args().collect();
    if argv.len() != 3 {
        eprintln!("Usage: {} <gguf-path> <tensor-name>", argv[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  {} /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/\\",
            argv[0]
        );
        eprintln!("    gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf \\");
        eprintln!("    v.blk.26.attn_q.weight");
        return ExitCode::from(2);
    }
    let gguf_path = Path::new(&argv[1]);
    let tensor_name = &argv[2];

    if !gguf_path.exists() {
        eprintln!("ERR: GGUF not found: {}", gguf_path.display());
        return ExitCode::from(2);
    }

    // -----------------------------------------------------------------
    // Production-path open + tensor info lookup.
    // -----------------------------------------------------------------
    let gguf = match GgufFile::open(gguf_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ERR: open {}: {}", gguf_path.display(), e);
            return ExitCode::from(2);
        }
    };

    let info = match gguf.tensor_info(tensor_name) {
        Some(t) => t.clone(),
        None => {
            eprintln!(
                "ERR: tensor '{}' not found in {}",
                tensor_name,
                gguf_path.display()
            );
            eprintln!("Available tensors (first 10):");
            for n in gguf.tensor_names().iter().take(10) {
                eprintln!("  {}", n);
            }
            return ExitCode::from(2);
        }
    };

    println!("== F16 weight byte-identity probe (ADR-005 iter 131) ==");
    println!("file:        {}", gguf_path.display());
    println!("tensor:      {}", tensor_name);
    println!("ggml_type:   {:?}", info.ggml_type);
    println!("shape:       {:?}", info.shape);
    println!("byte_len:    {}", info.byte_len);
    println!("rel_offset:  {}", info.offset);

    // -----------------------------------------------------------------
    // Path A — direct file read (bypasses GgufFile reader entirely).
    //
    // We can't read the absolute tensor_data_offset directly because it
    // is a private field on GgufFile. Workaround: walk to the start of
    // the data section by scanning forward from rel_offset 0 — i.e. we
    // re-derive the absolute offset by reading the bytes ONCE from the
    // GgufFile and then reading them AGAIN from a fresh File handle and
    // checking they match. This is sufficient: if the production reader
    // produces a different byte sequence than a fresh fd reading from
    // any consistent absolute offset, byte-identity is violated.
    //
    // Concretely: we use load_tensor's bytes as the reference (Path B)
    // and confirm with a fresh fd-read at the offset we *find* by
    // seeking the file to (file_len - tail_size) and walking back. The
    // simpler approach: load_tensor exposes the bytes in MlxBuffer
    // shared memory; we ALSO open the file fresh, seek-and-read by
    // matching the byte_len + a checksum-style first-bytes compare. If
    // bytes from a fresh fd at SOME offset that matches byte_len equal
    // the MlxBuffer storage, byte-identity is established.
    //
    // Sharper: since `tensor_data_offset` is private, we read the
    // entire file into memory ONCE (the mmproj is ~600 MB on disk for
    // the 27B Gemma 4V), find the byte_len sliding window that matches
    // the MlxBuffer's first 64 bytes uniquely, and compare. That's
    // expensive. Cheaper: rely on the public `GgufFile::load_tensor`
    // path being a memcpy from `read_tensor_bytes`, which is itself a
    // seek + read_exact. We compare two independent invocations of the
    // same path (calling load_tensor twice should produce byte-
    // identical results — sanity check), and we ALSO compare the
    // MlxBuffer storage against a hand-rolled GGUF header walk we
    // implement below.
    // -----------------------------------------------------------------

    // Simple, robust path: walk the GGUF header ourselves to compute
    // tensor_data_offset, then seek + read.
    let direct_bytes = match read_tensor_bytes_directly(gguf_path, tensor_name, &info) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERR: direct read: {e}");
            return ExitCode::from(2);
        }
    };
    if direct_bytes.len() != info.byte_len {
        eprintln!(
            "ERR: direct read returned {} bytes, expected {}",
            direct_bytes.len(),
            info.byte_len
        );
        return ExitCode::from(2);
    }

    // -----------------------------------------------------------------
    // Path B — production load_tensor → MlxBuffer storage.
    // -----------------------------------------------------------------
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ERR: MlxDevice::new: {e}");
            return ExitCode::from(2);
        }
    };
    let mlx_buf = match gguf.load_tensor(tensor_name, &device) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERR: load_tensor: {e}");
            return ExitCode::from(2);
        }
    };
    let buf_bytes: &[u8] = match mlx_buf.as_slice::<u8>() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERR: as_slice<u8>: {e}");
            return ExitCode::from(2);
        }
    };
    if buf_bytes.len() != info.byte_len {
        eprintln!(
            "ERR: MlxBuffer length {} != tensor byte_len {}",
            buf_bytes.len(),
            info.byte_len
        );
        return ExitCode::from(1);
    }

    // -----------------------------------------------------------------
    // Compare.
    // -----------------------------------------------------------------
    if direct_bytes == buf_bytes {
        println!("RESULT:      MATCH ({} bytes byte-identical)", info.byte_len);
        println!("falsifies:   W61 iter-130 candidate #2 — load path does NOT");
        println!("             corrupt F16 weights for this tensor.");
        return ExitCode::SUCCESS;
    }

    // Mismatch — find first divergent byte.
    let mut first = 0usize;
    while first < info.byte_len && direct_bytes[first] == buf_bytes[first] {
        first += 1;
    }
    println!(
        "RESULT:      MISMATCH at byte {} (direct=0x{:02x}, mlx=0x{:02x})",
        first, direct_bytes[first], buf_bytes[first]
    );
    let mut mismatches = 0usize;
    for i in 0..info.byte_len {
        if direct_bytes[i] != buf_bytes[i] {
            mismatches += 1;
        }
    }
    println!(
        "             {}/{} mismatched bytes ({:.4}% of tensor)",
        mismatches,
        info.byte_len,
        100.0 * (mismatches as f64) / (info.byte_len as f64)
    );
    ExitCode::from(1)
}

/// Read a single tensor's raw bytes from disk by walking the GGUF
/// header ourselves.
///
/// We deliberately do NOT use `mlx_native::gguf::GgufFile`'s reader for
/// this path so the result is independent. The on-disk layout is:
///
/// ```text
///   magic        : 4 bytes  "GGUF"
///   version      : u32
///   tensor_count : u64
///   metadata_kv_count : u64
///   metadata_kv      : variable
///   tensor_info[]    : variable
///   alignment_pad    : variable (to GENERAL_ALIGNMENT, default 32)
///   tensor_data      : tensors[i].byte_len each, padded
/// ```
///
/// We don't need to parse the metadata or tensor_info entries — we ALREADY
/// know the tensor's relative offset and byte_len from `info`. We only need
/// to find `tensor_data_offset`. The strategy: walk the header just enough
/// to skip past the metadata + tensor_info section, then align.
fn read_tensor_bytes_directly(
    gguf_path: &Path,
    _tensor_name: &str,
    info: &mlx_native::gguf::TensorInfo,
) -> Result<Vec<u8>, String> {
    let mut f = File::open(gguf_path).map_err(|e| format!("open: {e}"))?;
    let total_len = f
        .metadata()
        .map_err(|e| format!("metadata: {e}"))?
        .len();

    let tdo = parse_tensor_data_offset(&mut f)?;
    let abs_offset = tdo + info.offset;
    if abs_offset + (info.byte_len as u64) > total_len {
        return Err(format!(
            "computed abs_offset {} + byte_len {} > file len {}",
            abs_offset, info.byte_len, total_len
        ));
    }
    f.seek(SeekFrom::Start(abs_offset))
        .map_err(|e| format!("seek to {}: {e}", abs_offset))?;
    let mut buf = vec![0u8; info.byte_len];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read {} bytes at {}: {e}", info.byte_len, abs_offset))?;
    Ok(buf)
}

/// Parse just enough of the GGUF header to compute the tensor data
/// offset. Independent of the `mlx_native::gguf` parser so we can use
/// it as a cross-check.
fn parse_tensor_data_offset(f: &mut File) -> Result<u64, String> {
    f.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek 0: {e}"))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("read magic: {e}"))?;
    if &magic != b"GGUF" {
        return Err(format!("bad magic: {:?}", magic));
    }
    let _version = read_u32(f)?;
    let tensor_count = read_u64(f)?;
    let kv_count = read_u64(f)?;

    // Walk metadata.
    for _ in 0..kv_count {
        let _key = read_string(f)?;
        let val_type = read_u32(f)?;
        skip_value(f, val_type)?;
    }

    // Walk tensor info entries.
    for _ in 0..tensor_count {
        let _name = read_string(f)?;
        let n_dims = read_u32(f)?;
        for _ in 0..n_dims {
            let _ = read_u64(f)?;
        }
        let _ggml_type = read_u32(f)?;
        let _offset = read_u64(f)?;
    }

    // Align to GENERAL_ALIGNMENT (default 32). We don't know the exact
    // alignment from this side; mlx-native defaults to 32. The
    // alignment override is in metadata as
    // `general.alignment` — we trust the default for the gemma4 GGUFs
    // we ship; if the file uses a different alignment, the result will
    // mismatch and the user will see "MISMATCH at byte 0".
    let pos = f
        .stream_position()
        .map_err(|e| format!("stream_position: {e}"))?;
    let alignment: u64 = 32;
    let aligned = (pos + alignment - 1) / alignment * alignment;
    Ok(aligned)
}

fn read_u32(f: &mut File) -> Result<u32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(f: &mut File) -> Result<u64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)
        .map_err(|e| format!("read u64: {e}"))?;
    Ok(u64::from_le_bytes(b))
}
fn read_string(f: &mut File) -> Result<String, String> {
    let n = read_u64(f)? as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)
        .map_err(|e| format!("read string of {n}: {e}"))?;
    String::from_utf8(b).map_err(|e| format!("utf8: {e}"))
}

// GGUF metadata value types, mirroring the `MetadataValueType` enum in
// /opt/mlx-native/src/gguf/mod.rs. We only need to skip values, not
// interpret them.
const GGUF_TYPE_UINT8:   u32 = 0;
const GGUF_TYPE_INT8:    u32 = 1;
const GGUF_TYPE_UINT16:  u32 = 2;
const GGUF_TYPE_INT16:   u32 = 3;
const GGUF_TYPE_UINT32:  u32 = 4;
const GGUF_TYPE_INT32:   u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL:    u32 = 7;
const GGUF_TYPE_STRING:  u32 = 8;
const GGUF_TYPE_ARRAY:   u32 = 9;
const GGUF_TYPE_UINT64:  u32 = 10;
const GGUF_TYPE_INT64:   u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

fn skip_value(f: &mut File, t: u32) -> Result<(), String> {
    match t {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => {
            f.seek(SeekFrom::Current(1)).map_err(|e| format!("skip 1: {e}"))?;
        }
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => {
            f.seek(SeekFrom::Current(2)).map_err(|e| format!("skip 2: {e}"))?;
        }
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => {
            f.seek(SeekFrom::Current(4)).map_err(|e| format!("skip 4: {e}"))?;
        }
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => {
            f.seek(SeekFrom::Current(8)).map_err(|e| format!("skip 8: {e}"))?;
        }
        GGUF_TYPE_STRING => {
            let _ = read_string(f)?;
        }
        GGUF_TYPE_ARRAY => {
            let inner = read_u32(f)?;
            let n = read_u64(f)?;
            for _ in 0..n {
                skip_value(f, inner)?;
            }
        }
        other => {
            return Err(format!("unknown metadata type {other}"));
        }
    }
    Ok(())
}

