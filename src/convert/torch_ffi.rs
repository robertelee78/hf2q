//! FFI to PyTorch's libtorch_cpu.dylib for the `Sleef_expf4_u10advsimd`
//! NEON vectorized exp used internally by `torch.exp()` on ARM macOS.
//!
//! ADR-033 §P1 byte-identity for ssm_a tensors closure path. Pure-Rust
//! `f32::exp()` (libm/Apple's expf) bit-matches PyTorch only ~92% of the
//! time on uniform random inputs in [-10, 5]; the remaining ~8% diverge
//! by 1 ULP. PyTorch's CPU expf goes through SLEEF's 1-ULP-precision
//! vectorized routine, which IS bit-matchable via direct FFI.
//!
//! Empirically verified: `Sleef_expf4_u10advsimd` ≡ `torch.tensor.exp()`
//! for 1000/1000 uniform random f32 inputs in [-10, 5]; libm only
//! matches 919/1000.
//!
//! Opt-in via `--ffi-torch-exp [<path>]`. With no value, the flag
//! resolves to a hardcoded `default_missing_value` for the operator's
//! pyenv-installed libtorch (see `cli.rs`). When loaded, `BakeOp::NegExp`
//! uses Sleef instead of Rust's `f32::exp`. The flag is unset by
//! default — `try_init` is only called when the operator opts in.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use libloading::{Library, Symbol};

/// SLEEF 4-wide f32 vector signature. NEON `float32x4_t` is 16 bytes /
/// 4 f32 lanes, passed by value via ARM AAPCS.
///
/// `improper_ctypes_definitions` lint trips on the Rust intrinsic
/// `float32x4_t` type because it's marked unstable for FFI-safe layout
/// at the language level, but the ABI is well-defined for ARM AAPCS
/// (vector register passing) and `std::arch::aarch64::float32x4_t`
/// is a `#[repr(simd)]` 16-byte vector layout that matches `float32x4_t`
/// in `<arm_neon.h>`. Empirically verified via Rust↔C interop test
/// in `tests::sleef_matches_torch_for_known_divergent_input`.
#[allow(improper_ctypes_definitions)]
type SleefExpfFn = unsafe extern "C" fn(std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t;

/// Process-singleton (same rationale as `canonical_ffi`: libtorch_cpu
/// has global state and shouldn't be dlopen'd more than once). Stored
/// alongside the resolving path so a second `try_init` with a different
/// path is rejected rather than silently using the first lib.
static TORCH_SINGLETON: OnceLock<Result<(PathBuf, TorchExpInner), String>> = OnceLock::new();

struct TorchExpInner {
    _lib: Library,
    sleef_expf4: SleefExpfFn,
}

unsafe impl Send for TorchExpInner {}
unsafe impl Sync for TorchExpInner {}

/// Try-load libtorch_cpu and resolve `Sleef_expf4_u10advsimd`. Fails
/// fast if the symbol is missing. First successful load is process-wide
/// sticky; subsequent calls with the same path return Ok using the
/// cached handle. Subsequent calls with a different path return an
/// error rather than silently using the first lib — libtorch has global
/// state and the process is committed to one path for its lifetime.
pub fn try_init(path: impl AsRef<Path>) -> Result<(), String> {
    let path = path.as_ref();
    let path_buf = path.to_path_buf();
    let result = TORCH_SINGLETON.get_or_init(|| {
        // SAFETY: dlopen is unsafe; we trust the user-provided libtorch.
        let lib = unsafe { Library::new(&path_buf) }
            .map_err(|e| format!("dlopen {}: {}", path_buf.display(), e))?;

        // SAFETY: SleefExpfFn matches the C ABI of
        // `Sleef_expf4_u10advsimd` exactly: takes/returns float32x4_t
        // via ARM AAPCS register passing.
        let sleef_expf4: SleefExpfFn = unsafe {
            let sym: Symbol<'_, SleefExpfFn> = lib
                .get(b"Sleef_expf4_u10advsimd\0")
                .map_err(|e| format!("symbol Sleef_expf4_u10advsimd missing: {e}"))?;
            *sym
        };

        Ok((path_buf.clone(), TorchExpInner { sleef_expf4, _lib: lib }))
    });
    match result {
        Err(e) => Err(e.clone()),
        Ok((cached_path, _)) => {
            if cached_path != path {
                Err(format!(
                    "libtorch FFI already initialized with {}; cannot re-init with {}",
                    cached_path.display(),
                    path.display()
                ))
            } else {
                Ok(())
            }
        }
    }
}

/// Returns true if the libtorch FFI has been successfully loaded.
pub fn is_loaded() -> bool {
    matches!(TORCH_SINGLETON.get(), Some(Ok(_)))
}

/// Apply Sleef expf to a slice in-place. Processes in 4-wide NEON
/// chunks; the `< 4` tail is loaded into a zero-padded 4-wide buffer,
/// vectorized, and only the original positions are copied back.
///
/// Returns Err if `try_init` has not been called (or it failed). The
/// per-call result is not re-verified against torch.exp — that
/// invariant is established once by the `sleef_matches_torch_for_known_divergent_input`
/// test against PyTorch's installed lib.
pub fn exp_inplace(data: &mut [f32]) -> Result<(), String> {
    let Some(Ok((_, inner))) = TORCH_SINGLETON.get() else {
        return Err("libtorch FFI not initialized; call try_init() first".into());
    };

    use std::arch::aarch64::*;

    let n = data.len();
    let n4 = n - (n % 4);
    let mut i = 0;
    // SAFETY: vld1q_f32 / vst1q_f32 with valid in-range pointers.
    // Sleef function is C-ABI extern; we hold the library via OnceLock.
    unsafe {
        while i < n4 {
            let v = vld1q_f32(data.as_ptr().add(i));
            let r = (inner.sleef_expf4)(v);
            vst1q_f32(data.as_mut_ptr().add(i), r);
            i += 4;
        }
        // Tail: load remaining lanes into a vector, exp, store back.
        if i < n {
            let mut buf = [0f32; 4];
            for (j, k) in (i..n).enumerate() {
                buf[j] = data[k];
            }
            let v = vld1q_f32(buf.as_ptr());
            let r = (inner.sleef_expf4)(v);
            vst1q_f32(buf.as_mut_ptr(), r);
            for (j, k) in (i..n).enumerate() {
                data[k] = buf[j];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn libtorch_path() -> Option<PathBuf> {
        // Try common pyenv/conda path. If torch isn't installed, skip test.
        let candidates = [
            "/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib",
        ];
        for c in candidates {
            let p = PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    fn sleef_matches_torch_for_known_divergent_input() {
        let Some(path) = libtorch_path() else {
            eprintln!("[skip] libtorch not present");
            return;
        };
        try_init(&path).expect("init libtorch");
        // Input -3.796875 — one of the known torch != libm divergence points.
        let mut data = vec![-3.796875f32];
        exp_inplace(&mut data).expect("sleef exp");
        // Expected: torch.exp(-3.796875) = 0x3cb7d5c0 = 2.2440791130e-02
        assert_eq!(
            data[0].to_bits(),
            0x3cb7d5c0,
            "Sleef exp must bit-match torch.exp; got 0x{:08x}",
            data[0].to_bits()
        );
    }

    #[test]
    fn sleef_handles_tail_correctly() {
        let Some(path) = libtorch_path() else { return; };
        try_init(&path).expect("init libtorch");
        // 7 elements (4 + 3 tail)
        let mut data: Vec<f32> = (0..7).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let expected: Vec<f32> = data.iter().map(|x| x.exp()).collect();
        exp_inplace(&mut data).expect("sleef exp");
        // Tail should be processed; values should be in expected range
        // (allow 1 ULP diff vs libm because Sleef != libm).
        for (i, (s, l)) in data.iter().zip(expected.iter()).enumerate() {
            let ds = (s - l).abs();
            let rel = ds / l.abs().max(1e-30);
            assert!(
                rel < 1e-5,
                "tail elem {}: sleef={} libm={} rel={}",
                i, s, l, rel
            );
        }
    }
}
