//! GGUF v3 byte-format constants + KV-pair encoding utilities.
//!
//! Per ADR-033 §P2 + audit file `docs/adr-033-audit/gguf-writer.md`. The
//! KV-pair encoding logic here is byte-identical to the inline version
//! in `src/backends/gguf.rs::write_metadata_kv` (line 4007) and
//! `write_gguf_string` (line 3999) — existing GGUF files produced by
//! hf2q must round-trip through the new writer.
//!
//! GGUF v3 spec at `/opt/llama.cpp/ggml/include/ggml.h` and
//! `/opt/llama.cpp/gguf-py/gguf/constants.py`. The on-wire integer codes
//! for metadata-value types and ggml types are part of the spec; they
//! match llama.cpp at the SHA recorded in `data/llama_cpp_pin.txt`.

use std::io::{Result as IoResult, Write};

// ---------------------------------------------------------------------------
// Header magic + version + alignment
// ---------------------------------------------------------------------------

/// GGUF file magic: ASCII "GGUF" little-endian.
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// GGUF wire-format version emitted by hf2q (v3 — matches llama.cpp at
/// the pinned SHA).
pub const GGUF_VERSION: u32 = 3;

/// Tensor-data alignment in bytes. From llama.cpp `gguf_default_alignment`
/// at `ggml/src/gguf.cpp` (32 bytes). The header may also include an
/// explicit `general.alignment = <u32>` metadata key; v1 hf2q always
/// uses the default 32-byte alignment and does NOT emit the override.
pub const ALIGNMENT: u64 = 32;

// ---------------------------------------------------------------------------
// GGUF metadata-value type codes (on the wire, u32 LE)
// ---------------------------------------------------------------------------
//
// Mirror `enum gguf_metadata_value_type` at
// `/opt/llama.cpp/ggml/include/ggml.h` (search `GGUF_METADATA_VALUE_TYPE_`).

pub const GGUF_TYPE_UINT8: u32 = 0;
pub const GGUF_TYPE_INT8: u32 = 1;
pub const GGUF_TYPE_UINT16: u32 = 2;
pub const GGUF_TYPE_INT16: u32 = 3;
pub const GGUF_TYPE_UINT32: u32 = 4;
pub const GGUF_TYPE_INT32: u32 = 5;
pub const GGUF_TYPE_FLOAT32: u32 = 6;
pub const GGUF_TYPE_BOOL: u32 = 7;
pub const GGUF_TYPE_STRING: u32 = 8;
pub const GGUF_TYPE_ARRAY: u32 = 9;
pub const GGUF_TYPE_UINT64: u32 = 10;
pub const GGUF_TYPE_INT64: u32 = 11;
pub const GGUF_TYPE_FLOAT64: u32 = 12;

// ---------------------------------------------------------------------------
// MetaValue — typed GGUF metadata payload
// ---------------------------------------------------------------------------

/// Typed metadata value mirroring the GGUF v3 KV taxonomy.
///
/// The v1 set covers every variant emitted by the existing
/// `src/backends/gguf.rs::MetaValue` enum (private there; redefined
/// public here per ADR-033 P2's "extract" disposition). Variants are
/// ordered to match the on-wire type-code order for visual review.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrayU8(Vec<u8>),
    ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>),
    ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayBool(Vec<bool>),
    ArrayString(Vec<String>),
    ArrayU64(Vec<u64>),
    ArrayI64(Vec<i64>),
    ArrayF64(Vec<f64>),
}

// ---------------------------------------------------------------------------
// Wire-format encoding helpers
// ---------------------------------------------------------------------------

/// Encode a length-prefixed UTF-8 string: `u64 LE length | bytes`.
///
/// Matches `src/backends/gguf.rs:3999::write_gguf_string` byte-for-byte.
pub fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> IoResult<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

/// Encode a single KV pair: `string key | u32 type | <type-specific payload>`.
///
/// For arrays the payload is `u32 elem_type | u64 elem_count | elements`.
/// Byte-identical to `src/backends/gguf.rs:4007::write_metadata_kv` on
/// the overlap of variants the old enum exposed.
pub fn write_metadata_kv<W: Write>(w: &mut W, key: &str, value: &MetaValue) -> IoResult<()> {
    write_gguf_string(w, key)?;
    match value {
        MetaValue::U8(v) => {
            w.write_all(&GGUF_TYPE_UINT8.to_le_bytes())?;
            w.write_all(&[*v])?;
        }
        MetaValue::I8(v) => {
            w.write_all(&GGUF_TYPE_INT8.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::U16(v) => {
            w.write_all(&GGUF_TYPE_UINT16.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::I16(v) => {
            w.write_all(&GGUF_TYPE_INT16.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::U32(v) => {
            w.write_all(&GGUF_TYPE_UINT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::I32(v) => {
            w.write_all(&GGUF_TYPE_INT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::F32(v) => {
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::Bool(v) => {
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
        }
        MetaValue::String(s) => {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            write_gguf_string(w, s)?;
        }
        MetaValue::U64(v) => {
            w.write_all(&GGUF_TYPE_UINT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::I64(v) => {
            w.write_all(&GGUF_TYPE_INT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::F64(v) => {
            w.write_all(&GGUF_TYPE_FLOAT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::ArrayU8(arr) => write_array(w, GGUF_TYPE_UINT8, arr.len(), |w| {
            w.write_all(arr)
        })?,
        MetaValue::ArrayI8(arr) => write_array(w, GGUF_TYPE_INT8, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayU16(arr) => write_array(w, GGUF_TYPE_UINT16, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayI16(arr) => write_array(w, GGUF_TYPE_INT16, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayU32(arr) => write_array(w, GGUF_TYPE_UINT32, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayI32(arr) => write_array(w, GGUF_TYPE_INT32, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayF32(arr) => write_array(w, GGUF_TYPE_FLOAT32, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayBool(arr) => write_array(w, GGUF_TYPE_BOOL, arr.len(), |w| {
            for &b in arr {
                w.write_all(&[b as u8])?;
            }
            Ok(())
        })?,
        MetaValue::ArrayString(arr) => write_array(w, GGUF_TYPE_STRING, arr.len(), |w| {
            for s in arr {
                write_gguf_string(w, s)?;
            }
            Ok(())
        })?,
        MetaValue::ArrayU64(arr) => write_array(w, GGUF_TYPE_UINT64, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayI64(arr) => write_array(w, GGUF_TYPE_INT64, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
        MetaValue::ArrayF64(arr) => write_array(w, GGUF_TYPE_FLOAT64, arr.len(), |w| {
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        })?,
    }
    Ok(())
}

fn write_array<W: Write, F>(w: &mut W, elem_type: u32, count: usize, body: F) -> IoResult<()>
where
    F: FnOnce(&mut W) -> IoResult<()>,
{
    w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
    w.write_all(&elem_type.to_le_bytes())?;
    w.write_all(&(count as u64).to_le_bytes())?;
    body(w)
}

/// Round `offset` up to the next multiple of `alignment`.
///
/// Mirrors `src/backends/gguf.rs:4192::align_up` (and the inline impl in
/// `mlx-native::gguf::align_offset` at `/opt/mlx-native/src/gguf/mod.rs`).
#[inline]
pub fn align_up(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_spec() {
        assert_eq!(GGUF_MAGIC, *b"GGUF");
        assert_eq!(GGUF_VERSION, 3);
        assert_eq!(ALIGNMENT, 32);
    }

    #[test]
    fn write_string_format() {
        let mut buf = Vec::new();
        write_gguf_string(&mut buf, "hi").unwrap();
        // u64 LE length (2) + "hi"
        assert_eq!(
            buf,
            vec![2, 0, 0, 0, 0, 0, 0, 0, b'h', b'i']
        );
    }

    #[test]
    fn write_kv_u32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &MetaValue::U32(42)).unwrap();
        // "k" string (u64 1 + 'k') + type code (u32 4) + u32 LE 42
        let mut expected = Vec::new();
        expected.extend_from_slice(&1u64.to_le_bytes());
        expected.push(b'k');
        expected.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        expected.extend_from_slice(&42u32.to_le_bytes());
        assert_eq!(buf, expected);
    }

    #[test]
    fn write_kv_string() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "general.architecture", &MetaValue::String("llama".into()))
            .unwrap();
        // key u64-len + key + type=8 + val u64-len + val
        let mut expected = Vec::new();
        expected.extend_from_slice(&(20u64).to_le_bytes());
        expected.extend_from_slice(b"general.architecture");
        expected.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        expected.extend_from_slice(&(5u64).to_le_bytes());
        expected.extend_from_slice(b"llama");
        assert_eq!(buf, expected);
    }

    #[test]
    fn write_kv_array_u32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "arr", &MetaValue::ArrayU32(vec![1, 2, 3])).unwrap();
        let mut expected = Vec::new();
        // key
        expected.extend_from_slice(&3u64.to_le_bytes());
        expected.extend_from_slice(b"arr");
        // type=ARRAY, elem-type=UINT32, count=3, payload
        expected.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        expected.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        expected.extend_from_slice(&3u64.to_le_bytes());
        expected.extend_from_slice(&1u32.to_le_bytes());
        expected.extend_from_slice(&2u32.to_le_bytes());
        expected.extend_from_slice(&3u32.to_le_bytes());
        assert_eq!(buf, expected);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 32), 128);
    }
}
