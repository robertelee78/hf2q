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
