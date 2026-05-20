//! Pure-Rust port of SLEEF's `xexpf` (1-ULP f32 exponential) — the same
//! polynomial PyTorch's CPU `torch.exp` uses on ARM64 macOS via SLEEF.
//!
//! Source ported (BSL-1.0): `shibatch/sleef`
//!   - algorithm: `src/libm/sleefsimdsp.c:1314-1336` (function `xexpf`)
//!   - constants: `src/common/misc.h:115,141,142,144`
//!
//! Why: Rust's `f32::exp()` (libm/Apple expf) bit-matches PyTorch only
//! ~92% of the time on uniform random inputs in [-10, 5]; the remaining
//! ~8% diverge by 1 ULP because PyTorch routes through SLEEF's specific
//! polynomial. For ADR-033 §P1 byte-identity to canonical (which uses
//! `torch.exp` to bake `ssm_a = -exp(A_log)`), the bake side must use
//! the same polynomial.
//!
//! Per the operator's "we port -- we NEVER ffi" rule (2026-05-20):
//! the polynomial is implemented in pure Rust scalar arithmetic via
//! `f32::mul_add`, NOT via `libloading::dlopen` to libtorch/libsleef.
//! Verified bit-identical to torch.exp for the known ADR-033 divergent
//! input `-3.796875 → 0x3cb7d5c0`.

/// SLEEF Cody-Waite ln(2) split, high part (single-precision).
const L2UF: f32 = 0.693145751953125;

/// SLEEF Cody-Waite ln(2) split, low part (single-precision).
const L2LF: f32 = 1.428606765330187045e-06;

/// 1/ln(2) ≈ 1.4426950408... (truncated to f32 precision).
const R_LN2F: f32 = 1.442_695_040_888_963_4_f32;

/// SLEEF `xexpf` polynomial coefficients (Horner order, lowest-degree first).
/// Verbatim from `sleefsimdsp.c:1321-1326`.
const POLY0: f32 = 0.000_198_527_617_612_853_64;
const POLY1: f32 = 0.001_393_043_552_525_341_5;
const POLY2: f32 = 0.008_333_360_776_305_198_7;
const POLY3: f32 = 0.041_666_485_369_205_475;
const POLY4: f32 = 0.166_666_671_633_720_4;
const POLY5: f32 = 0.5;

/// Underflow/overflow thresholds (from sleefsimdsp.c:1332-1333).
const EXPF_UNDERFLOW: f32 = -104.0;
const EXPF_OVERFLOW: f32 = 100.0;

/// Pure-Rust port of SLEEF's `xexpf` (1-ULP precision exponential).
///
/// Algorithm:
///   1. Range reduction: `q = round(x / ln(2))`
///   2. Cody-Waite split: `s = x - q*L2Uf - q*L2Lf` (computed via fmadd)
///   3. Polynomial: 6-term Horner-form approximation of `(e^s - 1 - s) / s²`
///   4. Reconstruction: `e^x ≈ (1 + s + s²·p(s)) · 2^q`
///   5. Edge cases: `x < -104 → 0`; `x > 100 → +∞`
///
/// # Bit-identity contract
///
/// For all f32 inputs in `[-104, 100]`, this produces the SAME bit
/// pattern as PyTorch's `torch.tensor([x]).exp()` on ARM64 macOS.
/// Verified: input `-3.796875` produces `0x3cb7d5c0` (== `torch.exp`),
/// while libm produces `0x3cb7d5bf` (1-ULP off).
///
/// For inputs outside `[-104, 100]`, returns 0.0 or +∞ to match SLEEF.
/// NaN propagation matches the polynomial's natural behavior (NaN input
/// → NaN output through the Horner FMA chain).
#[inline]
pub fn sleef_expf(d: f32) -> f32 {
    // q = round(d / ln(2))  — IEEE-754 round-half-to-even
    let q_int = (d * R_LN2F).round_ties_even() as i32;
    let qf = q_int as f32;

    // Cody-Waite reduction: s = d - q*L2Uf, then s = s - q*L2Lf.
    // SLEEF uses `vmla(q, -L2Uf, d)` = `(-L2Uf) * q + d` = `d - q*L2Uf`
    // via FMA (single rounding). Match that with `f32::mul_add`.
    let s = qf.mul_add(-L2UF, d);
    let s = qf.mul_add(-L2LF, s);

    // Horner-form polynomial via FMA chain (matches SLEEF's `vmla` calls).
    let mut u: f32 = POLY0;
    u = u.mul_add(s, POLY1);
    u = u.mul_add(s, POLY2);
    u = u.mul_add(s, POLY3);
    u = u.mul_add(s, POLY4);
    u = u.mul_add(s, POLY5);

    // Reconstruct: e^s ≈ 1 + s + s²·p(s).
    // SLEEF: `u = 1.0 + vmla(s*s, u, s)` = `1 + (s*s)*u + s`.
    // Order matters for 1-ULP: the outer `+ 1.0` is the LAST op.
    let u = 1.0_f32 + (s * s).mul_add(u, s);

    // ldexp: e^x = e^s · 2^q via SLEEF's two-step split (`vldexp2_vf_vf_vi2`,
    // sleefsimdsp.c). For d in [-104, 100], q ranges over roughly
    // [-150, 144] — bigger than the f32 normal exponent window
    // [-126, 127]. Naive clamp truncates subnormal/near-overflow
    // regions (codex audit at 563b948b caught: `d = -100` → q = -144
    // should give a subnormal, clamp gave 2^-126). SLEEF avoids this
    // by splitting q two ways:
    //   q1 = q >> 1   (arithmetic shift; halves magnitude)
    //   q2 = q - q1   (remainder; q1 + q2 == q exactly)
    // then result = u · 2^q1 · 2^q2. Both halves stay within ±75 for
    // any q in [-150, 150], so bit-packing each via `(127+e) << 23` is
    // always safe and the product reaches the full subnormal/overflow
    // range correctly.
    let q1 = q_int >> 1;
    let q2 = q_int - q1;
    let two_q1 = f32::from_bits(((127_i32 + q1) as u32) << 23);
    let two_q2 = f32::from_bits(((127_i32 + q2) as u32) << 23);
    let y = u * two_q1 * two_q2;

    if d < EXPF_UNDERFLOW {
        0.0
    } else if d > EXPF_OVERFLOW {
        f32::INFINITY
    } else {
        y
    }
}

/// NEON 4-wide explicit-SIMD variant of [`sleef_expf`]. Processes
/// 4 f32 inputs in parallel via `std::arch::aarch64`. The scalar
/// fallback for the `< 4` tail uses the same `sleef_expf` so the
/// per-element bit-equivalence to torch is preserved.
///
/// Per [[feedback-we-port-never-ffi-2026-05-20]] this is a PURE-RUST
/// port — `std::arch::aarch64::*` are Rust intrinsics around the
/// ARM AAPCS NEON ABI, NOT FFI to a C library. The polynomial
/// coefficients are the same `POLY0..POLY5` constants used by the
/// scalar path (verbatim from SLEEF's `sleefsimdsp.c:1321-1326`).
///
/// FMA semantics: Rust's `vfmaq_f32(a, b, c)` returns `a + b*c`
/// (per `core::arch::aarch64` docs). SLEEF's `vmla_vf_vf_vf_vf`
/// returns `a*b + c`. So when porting `vmla(a, b, c)` we call
/// `vfmaq_f32(c, a, b)`.
///
/// Bit-equivalence to scalar `sleef_expf` is the contract: every
/// 4-wide lane must produce the same bits as scalar evaluation on
/// the same input. Verified by `neon_matches_scalar_on_sweep` test.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn sleef_expf_inplace_neon(data: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = data.len();
    let n4 = n - (n % 4);
    let mut i = 0;
    // SAFETY: pointer arithmetic stays within `data.as_mut_ptr()..+n4`,
    // all vld1q_f32 / vst1q_f32 read/write exactly 4 lanes. Intrinsics
    // are safe wrt aliasing because we only have one `&mut [f32]`.
    unsafe {
        while i < n4 {
            let v_d = vld1q_f32(data.as_ptr().add(i));

            // q_i = round_ties_even(d * R_LN2F)
            let v_q_f = vmulq_f32(v_d, vdupq_n_f32(R_LN2F));
            let v_q_i = vcvtnq_s32_f32(v_q_f);
            let v_qf = vcvtq_f32_s32(v_q_i);

            // Cody-Waite reduction: s = d - q*L2Uf - q*L2Lf.
            // SLEEF `vmla(q, -L2Uf, d)` = q*(-L2Uf) + d.
            // Rust `vfmaq_f32(d, q, neg_l2u)` = d + q*neg_l2u. Same.
            let v_neg_l2u = vdupq_n_f32(-L2UF);
            let v_s = vfmaq_f32(v_d, v_qf, v_neg_l2u);
            let v_neg_l2l = vdupq_n_f32(-L2LF);
            let v_s = vfmaq_f32(v_s, v_qf, v_neg_l2l);

            // Horner-form polynomial. SLEEF: `u = vmla(u, s, NEXT)` =
            // u*s + NEXT. Rust: `vfmaq_f32(NEXT, u, s)` = NEXT + u*s.
            let v_u = vdupq_n_f32(POLY0);
            let v_u = vfmaq_f32(vdupq_n_f32(POLY1), v_u, v_s);
            let v_u = vfmaq_f32(vdupq_n_f32(POLY2), v_u, v_s);
            let v_u = vfmaq_f32(vdupq_n_f32(POLY3), v_u, v_s);
            let v_u = vfmaq_f32(vdupq_n_f32(POLY4), v_u, v_s);
            let v_u = vfmaq_f32(vdupq_n_f32(POLY5), v_u, v_s);

            // u = 1.0 + (s*s) * u + s — order: outer +1.0 LAST.
            // SLEEF: `1.0 + vmla(s*s, u, s)` = 1.0 + (s*s)*u + s.
            // Rust: 1.0 + vfmaq_f32(s, s_sq, u) = 1.0 + (s + s_sq*u). Same algebraically AND same FMA order.
            let v_s_sq = vmulq_f32(v_s, v_s);
            let v_inner = vfmaq_f32(v_s, v_s_sq, v_u);
            let v_u = vaddq_f32(vdupq_n_f32(1.0), v_inner);

            // vldexp2 two-step split: 2^q = 2^q1 * 2^q2 where q1 = q>>1.
            let v_q1 = vshrq_n_s32(v_q_i, 1);
            let v_q2 = vsubq_s32(v_q_i, v_q1);
            let v_127 = vdupq_n_s32(127);
            let v_e1 = vshlq_n_s32(vaddq_s32(v_q1, v_127), 23);
            let v_e2 = vshlq_n_s32(vaddq_s32(v_q2, v_127), 23);
            let v_two_q1 = vreinterpretq_f32_s32(v_e1);
            let v_two_q2 = vreinterpretq_f32_s32(v_e2);
            let v_y = vmulq_f32(vmulq_f32(v_u, v_two_q1), v_two_q2);

            // Edge masks: d < -104 → 0; d > 100 → +inf.
            let v_neg104 = vdupq_n_f32(EXPF_UNDERFLOW);
            let v_p100 = vdupq_n_f32(EXPF_OVERFLOW);
            let v_zero = vdupq_n_f32(0.0);
            let v_inf = vreinterpretq_f32_u32(vdupq_n_u32(0x7f800000));
            let m_under = vcltq_f32(v_d, v_neg104);
            let m_over = vcltq_f32(v_p100, v_d);
            // vbslq_f32(mask, a, b): mask?a:b per lane.
            let v_y = vbslq_f32(m_under, v_zero, v_y);
            let v_y = vbslq_f32(m_over, v_inf, v_y);

            vst1q_f32(data.as_mut_ptr().add(i), v_y);
            i += 4;
        }
    }
    // Scalar tail (length < 4). Each element re-uses `sleef_expf` so
    // the bit-equivalence contract holds.
    while i < n {
        data[i] = sleef_expf(data[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact divergence point documented in ADR-033's old §P1 row:
    ///   input -3.796875 → torch.exp = 0x3cb7d5c0
    ///   libm.expf       = 0x3cb7d5bf  (1-ULP off)
    /// Pure-Rust sleef_expf MUST produce 0x3cb7d5c0.
    #[test]
    fn matches_torch_on_known_divergence_point() {
        let x = -3.796875_f32;
        let got = sleef_expf(x);
        let expected: u32 = 0x3cb7d5c0;
        assert_eq!(
            got.to_bits(),
            expected,
            "sleef_expf(-3.796875): expected 0x{:08x} (torch), got 0x{:08x}",
            expected,
            got.to_bits()
        );
    }

    /// libm and sleef_expf should differ on the divergence point —
    /// this asserts the port is doing something different from libm
    /// (not just a no-op copy of `f32::exp`).
    #[test]
    fn differs_from_libm_at_divergence_point() {
        let x = -3.796875_f32;
        let libm_bits = x.exp().to_bits();
        let sleef_bits = sleef_expf(x).to_bits();
        assert_ne!(
            libm_bits, sleef_bits,
            "libm and sleef_expf should differ at -3.796875 (1-ULP gap); both gave 0x{:08x}",
            libm_bits
        );
    }

    /// Standard reference cases — exp(0)=1, exp(1)≈e, exp(-1)≈1/e.
    /// Both libm and SLEEF must agree on these "easy" values.
    #[test]
    fn matches_standard_reference_values() {
        // exp(0) == 1.0 exactly
        assert_eq!(sleef_expf(0.0).to_bits(), 0x3f800000);
        // exp(1) ≈ 2.718281828... → 0x402df854
        assert_eq!(sleef_expf(1.0).to_bits(), 0x402df854);
        // exp(-1) ≈ 0.367879441... → 0x3ebc5ab2
        assert_eq!(sleef_expf(-1.0).to_bits(), 0x3ebc5ab2);
    }

    /// Edge cases per SLEEF: underflow → 0, overflow → +∞.
    #[test]
    fn underflow_and_overflow_thresholds() {
        assert_eq!(sleef_expf(-200.0), 0.0);
        assert_eq!(sleef_expf(200.0), f32::INFINITY);
        assert_eq!(sleef_expf(f32::NEG_INFINITY), 0.0);
        // NaN propagates through (input NaN → output NaN).
        assert!(sleef_expf(f32::NAN).is_nan());
    }

    /// Codex audit at commit `563b948b` caught a gap: the original
    /// `q_int.clamp(-126, 127)` truncated subnormal results for inputs
    /// like `d = -100` (q = -144). Now using SLEEF's two-step split
    /// `2^q = 2^q1 · 2^q2` with `q1 = q >> 1; q2 = q - q1` so subnormal
    /// and near-overflow ranges are correctly produced.
    ///
    /// Ground truth from `python3 -c "torch.tensor([X], dtype=torch.float32).exp()"`:
    ///   torch.exp(-100.0) = 3.78e-44 = 0x0000001b (subnormal)
    ///   torch.exp(-90.0)  = 8.19e-40 = 0x0008ec28 (subnormal)
    ///   torch.exp(  90.0) = +inf      = 0x7f800000 (overflow)
    #[test]
    fn matches_torch_in_subnormal_and_overflow_ranges() {
        // d = -100: q ≈ -144, requires vldexp2 split for correct subnormal.
        let r = sleef_expf(-100.0);
        assert_eq!(
            r.to_bits(),
            0x0000001b,
            "sleef_expf(-100): expected torch 0x0000001b (subnormal), got 0x{:08x}",
            r.to_bits()
        );

        // d = -90: q ≈ -129, just below normal range boundary.
        let r = sleef_expf(-90.0);
        assert_eq!(
            r.to_bits(),
            0x0008ec28,
            "sleef_expf(-90): expected torch 0x0008ec28 (subnormal), got 0x{:08x}",
            r.to_bits()
        );

        // d = 90: q ≈ 130, IEEE-overflows naturally to +inf during u * 2^q1 * 2^q2.
        let r = sleef_expf(90.0);
        assert_eq!(
            r.to_bits(),
            0x7f800000,
            "sleef_expf(90): expected torch +inf 0x7f800000, got 0x{:08x}",
            r.to_bits()
        );

        // d = -104 (exactly at the underflow guard): code computes y first, then
        // the explicit `d < -104` post-selection guard overrides to 0.0 (mirrors
        // SLEEF's `vsel_vf_vo_vf_vf` post-mask pattern in sleefsimdsp.c:1332).
        let r = sleef_expf(-104.0);
        assert_eq!(r, 0.0, "sleef_expf(-104) post-guard → 0");

        // d = 100 (exactly at the overflow guard): polynomial path produces a
        // finite f32 (torch.exp(100) ≈ 2.69e+43 ≈ infinity in f32, since f32 max ≈ 3.4e+38).
        // Sleef's explicit guard `d > 100 → +inf` is `>` not `>=`, so at d=100 we
        // take the polynomial path; verify it overflows naturally to +inf via IEEE
        // arithmetic in `u * 2^q1 * 2^q2`.
        let r = sleef_expf(100.0);
        assert_eq!(
            r.to_bits(),
            0x7f800000,
            "sleef_expf(100): expected +inf (natural IEEE overflow), got 0x{:08x}",
            r.to_bits()
        );
    }

    /// Pinned: the NEON 4-wide variant must produce bit-identical
    /// results to the scalar variant on the documented divergence
    /// point. Tests on a 4-wide buffer (the natural NEON unit) plus
    /// a 7-wide buffer (exercises the 4 SIMD + 3 scalar tail path).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_known_divergence_point() {
        let mut data4 = vec![-3.796875_f32; 4];
        sleef_expf_inplace_neon(&mut data4);
        for (i, &x) in data4.iter().enumerate() {
            assert_eq!(
                x.to_bits(),
                0x3cb7d5c0,
                "NEON lane {} mismatch: expected 0x3cb7d5c0 (torch), got 0x{:08x}",
                i,
                x.to_bits()
            );
        }

        // 7 elements: 4-wide SIMD + 3-element scalar tail.
        let mut data7: Vec<f32> = vec![
            -3.796875, 0.0, 1.0, -1.0, // first 4 → SIMD path
            -100.0, 90.0, -3.796875,    // last 3 → scalar tail
        ];
        sleef_expf_inplace_neon(&mut data7);
        let expected = [
            0x3cb7d5c0_u32, // -3.796875 → torch
            0x3f800000,     // exp(0) = 1
            0x402df854,     // exp(1) ≈ e
            0x3ebc5ab2,     // exp(-1) ≈ 1/e
            0x0000001b,     // exp(-100) subnormal
            0x7f800000,     // exp(90) → +inf
            0x3cb7d5c0,     // -3.796875 again (in tail)
        ];
        for (i, (&x, &e)) in data7.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                e,
                "lane {} (mixed SIMD/tail) mismatch: expected 0x{:08x}, got 0x{:08x}",
                i,
                e,
                x.to_bits()
            );
        }
    }

    /// Pinned: every input in the random sweep produces bit-equivalent
    /// output between scalar and NEON variants. If the NEON FMA order
    /// or polynomial coefficient ordering drifts, this catches it.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_sweep() {
        // Same deterministic mulberry32-ish sweep as
        // `sweep_shows_libm_divergence` but checking NEON ≡ scalar.
        let mut x: u32 = 0xCAFEBABE;
        const N: usize = 1024;
        let mut input: Vec<f32> = Vec::with_capacity(N);
        for _ in 0..N {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            input.push((x as f32 / u32::MAX as f32) * 15.0 - 10.0);
        }
        let mut scalar_out = vec![0.0_f32; N];
        for (i, &v) in input.iter().enumerate() {
            scalar_out[i] = sleef_expf(v);
        }
        let mut neon_out = input.clone();
        sleef_expf_inplace_neon(&mut neon_out);
        let mut differ = 0;
        for (i, (&s, &nu)) in scalar_out.iter().zip(neon_out.iter()).enumerate() {
            if s.to_bits() != nu.to_bits() {
                if differ < 5 {
                    eprintln!(
                        "  divergence at idx {}: input={}, scalar=0x{:08x}, neon=0x{:08x}",
                        i, input[i], s.to_bits(), nu.to_bits()
                    );
                }
                differ += 1;
            }
        }
        assert_eq!(
            differ, 0,
            "NEON sleef_expf must bit-match scalar on every input; {differ}/{N} differed"
        );
    }

    /// Microbench: time NEON `sleef_expf_inplace_neon` vs scalar
    /// `sleef_expf` loop vs libm `f32::exp` on 1M f32. Establishes
    /// the speedup from the 4-wide SIMD port — expected to close
    /// the ~20% libm cost gap measured at scalar.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn sleef_expf_neon_microbench() {
        const N: usize = 1_000_000;
        let mut data: Vec<f32> = Vec::with_capacity(N);
        let mut x: u32 = 0xCAFEBABE;
        for _ in 0..N {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            data.push((x as f32 / u32::MAX as f32) * 15.0 - 10.0);
        }
        const ITERS: usize = 5;

        // libm baseline.
        let mut libm_buf = vec![0f32; N];
        let mut libm_ns = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            for i in 0..N {
                libm_buf[i] = data[i].exp();
            }
            libm_ns.push(t0.elapsed().as_nanos() as u64);
            std::hint::black_box(&libm_buf);
        }
        libm_ns.sort();

        // Scalar sleef.
        let mut scalar_buf = vec![0f32; N];
        let mut scalar_ns = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            for i in 0..N {
                scalar_buf[i] = sleef_expf(data[i]);
            }
            scalar_ns.push(t0.elapsed().as_nanos() as u64);
            std::hint::black_box(&scalar_buf);
        }
        scalar_ns.sort();

        // NEON sleef.
        let mut neon_ns = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let mut neon_buf = data.clone();
            let t0 = std::time::Instant::now();
            sleef_expf_inplace_neon(&mut neon_buf);
            neon_ns.push(t0.elapsed().as_nanos() as u64);
            std::hint::black_box(&neon_buf);
        }
        neon_ns.sort();

        let libm_med = libm_ns[ITERS / 2] as f64;
        let scalar_med = scalar_ns[ITERS / 2] as f64;
        let neon_med = neon_ns[ITERS / 2] as f64;

        eprintln!(
            "\nsleef NEON microbench (N = {} f32, {} iters):\n  \
             libm  f32::exp:           {:.3} ms   {:.1} M elem/s    [1.00× ref]\n  \
             scalar sleef_expf:        {:.3} ms   {:.1} M elem/s    [{:.2}× vs libm]\n  \
             NEON sleef_expf_inplace:  {:.3} ms   {:.1} M elem/s    [{:.2}× vs libm]",
            N, ITERS,
            libm_med / 1e6, N as f64 / (libm_med / 1e9) / 1e6,
            scalar_med / 1e6, N as f64 / (scalar_med / 1e9) / 1e6, scalar_med / libm_med,
            neon_med / 1e6, N as f64 / (neon_med / 1e9) / 1e6, neon_med / libm_med,
        );
    }

    /// Microbench: time `sleef_expf` vs Rust's `f32::exp` (libm) on a
    /// 1M-element f32 buffer representative of a real ssm_a bake input.
    /// `#[ignore]`'d — run with:
    ///
    ///     cargo test --release --bin hf2q sleef_expf_microbench -- --ignored --nocapture
    ///
    /// Establishes that the pure-Rust SLEEF port doesn't impose a
    /// catastrophic perf cost vs libm — important since `BakeOp::NegExp`
    /// runs on every `ssm_a` tensor at convert time. Per ADR-036 the
    /// outer convert is rayon-parallel so this measurement is single-
    /// threaded kernel throughput.
    #[test]
    #[ignore]
    fn sleef_expf_microbench() {
        // Deterministic seed; matches the `sweep_shows_libm_divergence`
        // distribution so we measure on the same input domain as the
        // correctness sweep.
        const N: usize = 1_000_000;
        let mut data: Vec<f32> = Vec::with_capacity(N);
        let mut x: u32 = 0xCAFEBABE;
        for _ in 0..N {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            data.push((x as f32 / u32::MAX as f32) * 15.0 - 10.0);
        }

        const ITERS: usize = 5;

        // libm baseline.
        let mut libm_buf = vec![0f32; N];
        let mut libm_ns = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            for i in 0..N {
                libm_buf[i] = data[i].exp();
            }
            libm_ns.push(t0.elapsed().as_nanos() as u64);
            // Black-box: prevent the loop from being optimized away.
            std::hint::black_box(&libm_buf);
        }
        libm_ns.sort();
        let libm_median = libm_ns[ITERS / 2];

        // sleef_expf (the production path used by BakeOp::NegExp).
        let mut sleef_buf = vec![0f32; N];
        let mut sleef_ns = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            for i in 0..N {
                sleef_buf[i] = sleef_expf(data[i]);
            }
            sleef_ns.push(t0.elapsed().as_nanos() as u64);
            std::hint::black_box(&sleef_buf);
        }
        sleef_ns.sort();
        let sleef_median = sleef_ns[ITERS / 2];

        let libm_mhz = (N as f64 / (libm_median as f64 / 1e9)) / 1e6;
        let sleef_mhz = (N as f64 / (sleef_median as f64 / 1e9)) / 1e6;
        let ratio = sleef_median as f64 / libm_median as f64;

        eprintln!(
            "\nsleef_expf vs libm exp microbench (N = {} f32, {} iters):\n  \
             libm  f32::exp:     {:.3} ms median  ({:.1} M elem/s)\n  \
             sleef_expf (port):  {:.3} ms median  ({:.1} M elem/s)\n  \
             cost ratio:         {:.2}× (1.0 = same speed)\n  \
             libm all iters:    {:?}\n  \
             sleef all iters:   {:?}",
            N,
            ITERS,
            libm_median as f64 / 1e6,
            libm_mhz,
            sleef_median as f64 / 1e6,
            sleef_mhz,
            ratio,
            libm_ns,
            sleef_ns,
        );
    }

    /// Sweep test: across 1024 uniform-random-ish f32 inputs in [-10, 5],
    /// sleef_expf should NEVER agree with libm 100% of the time (we know
    /// from empirical measurement ~92% agree, ~8% diverge by 1-ULP).
    /// This asserts the port is producing the SLEEF-specific polynomial
    /// values, not just shadowing libm.
    #[test]
    fn sweep_shows_libm_divergence() {
        // Deterministic pseudo-random sequence via xorshift.
        let mut x: u32 = 0xCAFEBABE;
        let mut differ = 0usize;
        let mut total = 0usize;
        for _ in 0..1024 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // Map u32 → [-10, 5] uniformly.
            let f = (x as f32 / u32::MAX as f32) * 15.0 - 10.0;
            let lb = f.exp().to_bits();
            let sb = sleef_expf(f).to_bits();
            if lb != sb {
                differ += 1;
            }
            total += 1;
        }
        // Empirically ~8% diverge. Assert at least 2% to leave margin.
        assert!(
            differ * 100 >= total * 2,
            "sleef_expf appears to shadow libm: only {}/{} diverged (expected ~8%)",
            differ,
            total
        );
    }
}
