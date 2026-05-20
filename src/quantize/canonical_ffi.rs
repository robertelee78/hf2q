//! FFI to canonical `libggml-base.0.dylib` quantize functions.
//!
//! ADR-033 §P1 byte-identity closure path. Opt-in via `--ffi-canonical`.
//! When activated, hf2q dispatches per-tensor quantization through
//! canonical's exported `quantize_q*` / `quantize_iq*` symbols instead
//! of the pure-Rust kernels under `ggml_quants/`. The tensor-type
//! POLICY remains in hf2q (`StandardPolicy::target_for`); only the
//! kernel CALL is swapped.
//!
//! Rationale: pure-Rust kernels diverge from canonical at sub-block
//! 1-ULP boundary cases due to clang's auto-vectorized NEON reduction
//! order vs Rust scalar `.mul_add`. The divergence is input-distribution
//! dependent — fixing one boundary case can open another, and there is
//! no clean source-level Rust pattern that matches all canonical
//! emit variants. FFI delegates to canonical's exact compiled emit
//! and is the only path to guaranteed byte-identity across all input
//! distributions.
//!
//! Pinned to `/opt/llama.cpp` HEAD `e15384a5cb` (matches
//! `data/llama_cpp_pin.txt`). The library ABI is NOT guaranteed
//! stable across llama.cpp SHAs — this loader fails fast if any
//! expected symbol is missing.

use std::path::Path;
use std::sync::OnceLock;
use libloading::{Library, Symbol};

use super::ggml_quants::{GgmlType, QuantizeError};

/// Process-wide canonical FFI singleton. ggml-base.0.dylib's static
/// initializer (`ggml_op_can_inplace` + uncaught_exception state) does
/// not tolerate being `dlopen`'d more than once in the same process —
/// the second open triggers `GGML_ASSERT(prev != ggml_uncaught_exception)`
/// in ggml.cpp's TLS init path. To stay safe we cache the loaded
/// Library + resolved symbols per-process via OnceLock.
///
/// `Result<...>` is stored so a failed load is sticky for the run
/// (subsequent requests get the same error). Path equality is checked
/// at access time — if the operator passes a different path on a
/// second call, we error out rather than silently returning the cached
/// handle.
static FFI_SINGLETON: OnceLock<Result<(std::path::PathBuf, CanonicalQuantInner), String>> = OnceLock::new();

/// C signature mirrors `/opt/llama.cpp/ggml/include/ggml.h`:
///
/// ```c
/// size_t quantize_q4_K(const float * src, void * dst,
///                      int64_t nrow, int64_t n_per_row,
///                      const float * imatrix);
/// ```
///
/// Returns bytes written. `imatrix` may be NULL (no per-row weights).
type QuantFn = unsafe extern "C" fn(
    src: *const f32,
    dst: *mut libc::c_void,
    nrow: i64,
    n_per_row: i64,
    imatrix: *const f32,
) -> usize;

/// Inner table of resolved function pointers + the owning Library. Owned
/// once-per-process by `FFI_SINGLETON`. Function pointers are valid for
/// the lifetime of the Library, which itself is `'static` once stored.
///
/// Marked `Send + Sync` manually — function pointers are FFI-safe and
/// libloading::Library is Send. Sync requires that the underlying
/// canonical lib's exported functions are thread-safe. canonical's
/// `quantize_q*` / `quantize_iq*` are stateless per-call (no global
/// state mutation per ggml-quants.c), so concurrent calls from different
/// threads are safe.
struct CanonicalQuantInner {
    _lib: Library,
    q2_k: QuantFn,
    q3_k: QuantFn,
    q4_k: QuantFn,
    q5_k: QuantFn,
    q6_k: QuantFn,
    q4_0: QuantFn,
    q4_1: QuantFn,
    q5_0: QuantFn,
    q5_1: QuantFn,
    q8_0: QuantFn,
    iq4_nl: QuantFn,
    iq4_xs: QuantFn,
}

// SAFETY: Library is Send. Function pointers are Send+Sync. canonical's
// exported quantize_* functions are per-call stateless.
unsafe impl Send for CanonicalQuantInner {}
unsafe impl Sync for CanonicalQuantInner {}

/// FFI handle to canonical libggml-base. Holds a `&'static` reference
/// to the process-singleton inner table. Cheap to clone.
pub struct CanonicalQuant {
    inner: &'static CanonicalQuantInner,
}

impl CanonicalQuant {
    /// Default canonical library path (matches the pinned llama.cpp build).
    pub const DEFAULT_PATH: &'static str = "/opt/llama.cpp/build/bin/libggml-base.0.dylib";

    /// Load the canonical library (process-wide singleton — first call
    /// resolves symbols, subsequent calls return a handle to the same
    /// loaded library). Errors are sticky: if the first call fails, all
    /// subsequent calls return the same error string.
    ///
    /// Passing a path different from the first call's path returns
    /// `FfiError::PathMismatch` rather than silently using the cached
    /// handle — canonical's static initializer can't tolerate a second
    /// dlopen, so the process is committed to one path for its lifetime.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FfiError> {
        let path = path.as_ref();
        let path_buf = path.to_path_buf();

        let entry = FFI_SINGLETON.get_or_init(|| {
            // SAFETY: `Library::new` is unsafe because loading a dylib can
            // execute arbitrary code via initializers. We trust
            // /opt/llama.cpp's libggml-base.
            let lib = unsafe { Library::new(&path_buf) }.map_err(|e| {
                format!("dlopen {}: {}", path_buf.display(), e)
            })?;

            // SAFETY: All symbols are extern "C" functions matching the C
            // signature in ggml.h. We copy the function pointer value;
            // the library lives for the static lifetime.
            unsafe fn resolve(lib: &Library, name: &[u8]) -> Result<QuantFn, String> {
                let sym: Symbol<'_, QuantFn> = lib.get(name).map_err(|e| {
                    format!(
                        "symbol `{}` missing: {}",
                        String::from_utf8_lossy(&name[..name.len() - 1]),
                        e
                    )
                })?;
                Ok(*sym)
            }

            let inner = unsafe {
                CanonicalQuantInner {
                    q2_k: resolve(&lib, b"quantize_q2_K\0")?,
                    q3_k: resolve(&lib, b"quantize_q3_K\0")?,
                    q4_k: resolve(&lib, b"quantize_q4_K\0")?,
                    q5_k: resolve(&lib, b"quantize_q5_K\0")?,
                    q6_k: resolve(&lib, b"quantize_q6_K\0")?,
                    q4_0: resolve(&lib, b"quantize_q4_0\0")?,
                    q4_1: resolve(&lib, b"quantize_q4_1\0")?,
                    q5_0: resolve(&lib, b"quantize_q5_0\0")?,
                    q5_1: resolve(&lib, b"quantize_q5_1\0")?,
                    q8_0: resolve(&lib, b"quantize_q8_0\0")?,
                    iq4_nl: resolve(&lib, b"quantize_iq4_nl\0")?,
                    iq4_xs: resolve(&lib, b"quantize_iq4_xs\0")?,
                    _lib: lib,
                }
            };
            Ok((path_buf.clone(), inner))
        });

        match entry {
            Err(e) => Err(FfiError::LoadFailed {
                path: path.display().to_string(),
                detail: e.clone(),
            }),
            Ok((cached_path, inner)) => {
                if cached_path != path {
                    return Err(FfiError::PathMismatch {
                        first: cached_path.display().to_string(),
                        requested: path.display().to_string(),
                    });
                }
                Ok(Self { inner })
            }
        }
    }

    /// Quantize via canonical FFI. `src` is `n_per_row * nrow` F32
    /// values (row-major); `dst` must be sized for the target type
    /// (caller computes via `GgmlType::row_size`). `imatrix` is
    /// optional per-row weights.
    ///
    /// Internally rayon-parallelized by row chunks: each chunk calls
    /// canonical's `quantize_q*` on its slice of rows. Per-row
    /// independence is structural (each block's quantization uses only
    /// that block's inputs and the per-row imatrix slice; no cross-row
    /// state). Output bytes are bit-identical to a single sequential
    /// call.
    pub fn quantize(
        &self,
        ggml_type: GgmlType,
        src: &[f32],
        n_per_row: usize,
        imatrix: Option<&[f32]>,
        dst: &mut [u8],
    ) -> Result<(), FfiError> {
        use rayon::prelude::*;

        let func: QuantFn = match ggml_type {
            GgmlType::Q2_K => self.inner.q2_k,
            GgmlType::Q3_K => self.inner.q3_k,
            GgmlType::Q4_K => self.inner.q4_k,
            GgmlType::Q5_K => self.inner.q5_k,
            GgmlType::Q6_K => self.inner.q6_k,
            GgmlType::Q4_0 => self.inner.q4_0,
            GgmlType::Q4_1 => self.inner.q4_1,
            GgmlType::Q5_0 => self.inner.q5_0,
            GgmlType::Q5_1 => self.inner.q5_1,
            GgmlType::Q8_0 => self.inner.q8_0,
            GgmlType::IQ4_NL => self.inner.iq4_nl,
            GgmlType::IQ4_XS => self.inner.iq4_xs,
            _ => return Err(FfiError::UnsupportedType { ggml_type }),
        };

        if src.len() % n_per_row != 0 {
            return Err(FfiError::ShapeMismatch {
                src_len: src.len(),
                n_per_row,
            });
        }
        let nrow = src.len() / n_per_row;
        let row_size = ggml_type.row_size(n_per_row);
        if dst.len() != nrow * row_size {
            return Err(FfiError::WriteSizeMismatch {
                expected: nrow * row_size,
                actual: dst.len(),
            });
        }

        // Single-row fast path: avoid rayon overhead for tiny tensors.
        if nrow == 1 {
            let imatrix_ptr = imatrix.map(|m| m.as_ptr()).unwrap_or(std::ptr::null());
            let wrote = unsafe {
                func(src.as_ptr(), dst.as_mut_ptr().cast(), 1, n_per_row as i64, imatrix_ptr)
            };
            if wrote != dst.len() {
                return Err(FfiError::WriteSizeMismatch {
                    expected: dst.len(),
                    actual: wrote,
                });
            }
            return Ok(());
        }

        // Parallel path: split into ~16 rayon chunks (or 1 row per chunk
        // if nrow < 16). Each chunk calls canonical FFI on its row slice.
        // Per-row independence guarantees the concatenated output is
        // bit-identical to a single sequential call.
        let n_chunks = nrow.min(16);
        let rows_per_chunk = nrow.div_ceil(n_chunks);

        // Build (src_slice, dst_slice) pairs per chunk. The unsafe
        // dst splitting via raw pointers is required because rayon
        // needs &mut [u8] across non-overlapping ranges, but
        // chunks_mut won't give arbitrary stride lengths in all-but-last
        // case. We use `split_at_mut` iteratively to enforce non-overlap.
        let row_chunks: Vec<(usize, usize)> = (0..n_chunks)
            .map(|i| {
                let start = i * rows_per_chunk;
                let end = (start + rows_per_chunk).min(nrow);
                (start, end)
            })
            .filter(|(s, e)| s < e)
            .collect();

        // Collect chunk results into Vec<u8> per chunk, then copy back
        // into the dst slice. This avoids unsafe pointer juggling at
        // the cost of one extra allocation per chunk (acceptable —
        // dst is row-sized, not full-tensor).
        let chunk_payloads: Result<Vec<(usize, Vec<u8>)>, FfiError> = row_chunks
            .par_iter()
            .map(|&(start, end)| {
                let chunk_rows = end - start;
                let chunk_src = &src[start * n_per_row..end * n_per_row];
                let chunk_im = imatrix; // per-row weights reused per row
                let chunk_imatrix_ptr = chunk_im.map(|m| m.as_ptr()).unwrap_or(std::ptr::null());
                let mut chunk_dst = vec![0u8; chunk_rows * row_size];

                let wrote = unsafe {
                    func(
                        chunk_src.as_ptr(),
                        chunk_dst.as_mut_ptr().cast(),
                        chunk_rows as i64,
                        n_per_row as i64,
                        chunk_imatrix_ptr,
                    )
                };
                if wrote != chunk_dst.len() {
                    return Err(FfiError::WriteSizeMismatch {
                        expected: chunk_dst.len(),
                        actual: wrote,
                    });
                }
                Ok((start, chunk_dst))
            })
            .collect();

        let chunks = chunk_payloads?;
        for (start, payload) in chunks {
            let dst_off = start * row_size;
            dst[dst_off..dst_off + payload.len()].copy_from_slice(&payload);
        }
        Ok(())
    }
}

/// FFI errors. Surfaced typed (no fallback to pure-Rust kernel) so any
/// canonical-FFI failure is immediately visible to the operator.
#[derive(Debug, thiserror::Error)]
pub enum FfiError {
    #[error("failed to dlopen canonical libggml at {path}: {detail}")]
    LoadFailed { path: String, detail: String },

    #[error("canonical libggml symbol `{name}` missing: {detail}")]
    SymbolMissing { name: String, detail: String },

    #[error(
        "canonical libggml already loaded from `{first}`; cannot reload from \
         `{requested}` — ggml-base's static initializer panics on a second \
         dlopen in the same process. Use a consistent path across all FFI calls."
    )]
    PathMismatch { first: String, requested: String },

    #[error(
        "canonical FFI does not support ggml_type {ggml_type:?} \
         (only Q*_K, Q*_0, Q*_1, IQ4_NL, IQ4_XS exported)"
    )]
    UnsupportedType { ggml_type: GgmlType },

    #[error("src len {src_len} not divisible by n_per_row {n_per_row}")]
    ShapeMismatch { src_len: usize, n_per_row: usize },

    #[error(
        "canonical quantize wrote {actual} bytes, expected {expected} \
         (caller miscomputed dst buffer size or ABI mismatch)"
    )]
    WriteSizeMismatch { expected: usize, actual: usize },
}

// Compile-time check that the inner `source` strings render without
// the broken `as_dyn_error` machinery — they're already `Display`.
// (thiserror tries to call `.source().as_dyn_error()` for String when
// the field name is `source`; rename or override would also work.)
// The `{source}` placeholder we used above implies `Display` from the
// field, which String supports natively. The error chain doesn't
// expose String as a source. To silence thiserror's heuristic, mark
// the field with `#[source]` semantics OFF — which is the default
// when the field is named differently. We work around by overriding:

impl From<FfiError> for QuantizeError {
    fn from(e: FfiError) -> Self {
        QuantizeError::CanonicalFfi(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn canonical_path() -> Option<PathBuf> {
        let p = PathBuf::from(CanonicalQuant::DEFAULT_PATH);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn loads_when_canonical_present() {
        let Some(path) = canonical_path() else {
            eprintln!("[skip] canonical libggml not present at {}", CanonicalQuant::DEFAULT_PATH);
            return;
        };
        let ffi = CanonicalQuant::load(&path).expect("load canonical");
        // Sanity: quantize a single Q8_0 block (32 floats → 34 bytes).
        let src = vec![0.5f32; 32];
        let mut dst = vec![0u8; 34];
        ffi.quantize(GgmlType::Q8_0, &src, 32, None, &mut dst)
            .expect("quantize_q8_0 via FFI");
        // First 2 bytes = F16 scale d, next 32 bytes = i8 quants.
        // All-0.5 input → all-same-sign quants; just verify non-zero output.
        assert!(dst.iter().any(|&b| b != 0), "FFI produced all-zero Q8_0 output");
    }

    #[test]
    fn rejects_unsupported_type() {
        let Some(path) = canonical_path() else { return; };
        let ffi = CanonicalQuant::load(&path).unwrap();
        let src = vec![1.0f32; 32];
        let mut dst = vec![0u8; 64];
        // F16 is not in the exported quantize_* set.
        let err = ffi.quantize(GgmlType::F16, &src, 32, None, &mut dst);
        assert!(matches!(err, Err(FfiError::UnsupportedType { .. })));
    }
}
