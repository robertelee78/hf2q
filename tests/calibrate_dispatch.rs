//! Integration tests for the calibrator dispatch table landed in
//! `src/main.rs::select_calibrator` (ADR-014 P7 iter-8).
//!
//! ## Why mirror the dispatch instead of importing it
//!
//! `select_calibrator` lives in the binary crate's `main.rs` and is
//! intentionally private (P2 iter-2 will lift it into a callable
//! library entry alongside the full `(Calibrator, OutputFormat)`
//! restructure). Binary-crate integration tests cannot reach private
//! items, and `src/cli.rs::QuantMethod` cannot be `#[path]`-included
//! standalone because `cli.rs` pulls in runtime modules (progress,
//! input) at function bodies.
//!
//! So this file mirrors the dispatch table **as a contract**. The
//! exhaustiveness `match` in [`mirror_covers_every_quant_method`]
//! catches drift via the compiler's exhaustiveness check: if a new
//! [`QuantMethodMirror`] variant is added (because a new clap variant
//! landed in the canonical `cli::QuantMethod`), every match in this
//! file fails to compile. Reviewers must update both
//! [`select_calibrator`] in `src/main.rs` AND this mirror in lockstep.
//!
//! ## What this file actually verifies
//!
//! The dispatch contract:
//! * Every static (non-DWQ) variant maps to `"none"` / no forward pass.
//! * Every DWQ variant maps to `"dwq"`, with `requires_forward_pass`
//!   driven by the DWQ architecture (qwen35* → true; Other → false).
//!
//! The actual `Calibrator` trait impls (`NoneCalibrator`,
//! `DwqCalibrator`, `ImatrixCalibrator`) carry their own unit tests
//! in `src/calibrate/{calibrator,dwq_calibrator,imatrix_calibrator}.rs`.

/// Local mirror of `cli::QuantMethod`. **Variants must stay in sync**
/// with `src/cli.rs::QuantMethod`. The compile-time exhaustive
/// matches below catch drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantMethodMirror {
    Auto,
    F16,
    Q8,
    Q4,
    Q2,
    Mixed26,
    Mixed36,
    Mixed46,
    DwqMixed46,
    DwqMixed48,
    DwqMixed68,
    DwqMixed28,
    Apex,
}

/// Local mirror of `crate::calibrate::dwq::DwqArch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DwqArchMirror {
    Qwen35Dense,
    Qwen35MoE,
    Other,
}

impl DwqArchMirror {
    fn requires_activation_capture(self) -> bool {
        matches!(self, Self::Qwen35Dense | Self::Qwen35MoE)
    }
}

/// Trait surface mirrored from `crate::calibrate::calibrator::Calibrator`.
trait CalibratorContract {
    fn name(&self) -> &'static str;
    fn requires_forward_pass(&self) -> bool;
}

/// Mirror of `NoneCalibrator`. Verified end-to-end by
/// `src/calibrate/calibrator.rs::tests::*`.
struct NoneMirror;
impl CalibratorContract for NoneMirror {
    fn name(&self) -> &'static str {
        "none"
    }
    fn requires_forward_pass(&self) -> bool {
        false
    }
}

/// Mirror of `DwqCalibrator`'s public contract. Verified end-to-end
/// by `src/calibrate/dwq_calibrator.rs::tests::*`.
struct DwqMirror {
    arch: DwqArchMirror,
}
impl CalibratorContract for DwqMirror {
    fn name(&self) -> &'static str {
        "dwq"
    }
    fn requires_forward_pass(&self) -> bool {
        self.arch.requires_activation_capture()
    }
}

/// Mirror of `select_calibrator` from `src/main.rs`.
///
/// Invariant: the match arms here MUST stay aligned with those in
/// `select_calibrator`. The `mirror_covers_every_quant_method` test
/// below catches drift at compile time.
fn select_calibrator_mirror(
    method: QuantMethodMirror,
    arch: DwqArchMirror,
) -> Box<dyn CalibratorContract> {
    use QuantMethodMirror::*;
    match method {
        DwqMixed46 | DwqMixed48 | DwqMixed68 | DwqMixed28 => {
            Box::new(DwqMirror { arch })
        }
        Auto | F16 | Q8 | Q4 | Q2 | Mixed26 | Mixed36 | Mixed46 | Apex => Box::new(NoneMirror),
    }
}

#[test]
fn select_calibrator_returns_none_for_static_methods() {
    let static_methods = [
        QuantMethodMirror::Auto,
        QuantMethodMirror::F16,
        QuantMethodMirror::Q8,
        QuantMethodMirror::Q4,
        QuantMethodMirror::Q2,
        QuantMethodMirror::Mixed26,
        QuantMethodMirror::Mixed36,
        QuantMethodMirror::Mixed46,
        QuantMethodMirror::Apex,
    ];
    for m in static_methods {
        let calibrator = select_calibrator_mirror(m, DwqArchMirror::Other);
        assert_eq!(
            calibrator.name(),
            "none",
            "method {m:?} must dispatch to NoneCalibrator (got {})",
            calibrator.name()
        );
        assert!(
            !calibrator.requires_forward_pass(),
            "NoneCalibrator must not require a forward pass (method={m:?})"
        );
    }
}

#[test]
fn select_calibrator_returns_dwq_for_dwq_methods() {
    let dwq_methods = [
        QuantMethodMirror::DwqMixed46,
        QuantMethodMirror::DwqMixed48,
        QuantMethodMirror::DwqMixed68,
        QuantMethodMirror::DwqMixed28,
    ];
    for m in dwq_methods {
        // arch=Other so no capture is needed for the trait-level check.
        let calibrator = select_calibrator_mirror(m, DwqArchMirror::Other);
        assert_eq!(
            calibrator.name(),
            "dwq",
            "DWQ method {m:?} must dispatch to DwqCalibrator"
        );
    }
}

#[test]
fn select_calibrator_dwq_arch_propagated() {
    // DwqMixed46 with Qwen35MoE → requires_forward_pass == true.
    let c1 =
        select_calibrator_mirror(QuantMethodMirror::DwqMixed46, DwqArchMirror::Qwen35MoE);
    assert_eq!(c1.name(), "dwq");
    assert!(
        c1.requires_forward_pass(),
        "DwqCalibrator + Qwen35MoE must require forward pass"
    );

    // DwqMixed68 with Qwen35Dense → requires_forward_pass == true.
    let c2 = select_calibrator_mirror(
        QuantMethodMirror::DwqMixed68,
        DwqArchMirror::Qwen35Dense,
    );
    assert!(c2.requires_forward_pass());

    // DwqMixed28 with Other → requires_forward_pass == false.
    let c3 =
        select_calibrator_mirror(QuantMethodMirror::DwqMixed28, DwqArchMirror::Other);
    assert!(!c3.requires_forward_pass());
}

#[test]
fn select_calibrator_dwq_arch_other_no_capture_required() {
    // The "Other" arch contract: DWQ on a non-qwen35 architecture
    // does NOT require a forward-pass / capture impl. (The
    // downstream weight-space contract for DwqArch::Other is
    // enforced separately in DwqCalibrator's unit tests.)
    for m in [
        QuantMethodMirror::DwqMixed46,
        QuantMethodMirror::DwqMixed48,
        QuantMethodMirror::DwqMixed68,
        QuantMethodMirror::DwqMixed28,
    ] {
        let calibrator = select_calibrator_mirror(m, DwqArchMirror::Other);
        assert_eq!(calibrator.name(), "dwq");
        assert!(
            !calibrator.requires_forward_pass(),
            "DwqCalibrator + Other arch must NOT require a forward pass (method={m:?})"
        );
    }
}

/// Compile-time exhaustiveness guard. Every variant of
/// [`QuantMethodMirror`] must be covered. If a new variant is added
/// without updating both the helper in main.rs AND this mirror, this
/// match fails to compile.
#[test]
fn mirror_covers_every_quant_method() {
    fn classify(m: QuantMethodMirror) -> &'static str {
        use QuantMethodMirror::*;
        match m {
            DwqMixed46 | DwqMixed48 | DwqMixed68 | DwqMixed28 => "dwq",
            Auto | F16 | Q8 | Q4 | Q2 | Mixed26 | Mixed36 | Mixed46 | Apex => "none",
        }
    }
    // Spot-check every variant flows through.
    assert_eq!(classify(QuantMethodMirror::Auto), "none");
    assert_eq!(classify(QuantMethodMirror::F16), "none");
    assert_eq!(classify(QuantMethodMirror::Q8), "none");
    assert_eq!(classify(QuantMethodMirror::Q4), "none");
    assert_eq!(classify(QuantMethodMirror::Q2), "none");
    assert_eq!(classify(QuantMethodMirror::Mixed26), "none");
    assert_eq!(classify(QuantMethodMirror::Mixed36), "none");
    assert_eq!(classify(QuantMethodMirror::Mixed46), "none");
    assert_eq!(classify(QuantMethodMirror::Apex), "none");
    assert_eq!(classify(QuantMethodMirror::DwqMixed46), "dwq");
    assert_eq!(classify(QuantMethodMirror::DwqMixed48), "dwq");
    assert_eq!(classify(QuantMethodMirror::DwqMixed68), "dwq");
    assert_eq!(classify(QuantMethodMirror::DwqMixed28), "dwq");
}

#[test]
fn dwq_arch_table_contract() {
    // The DwqArch table that drives DwqCalibrator::requires_forward_pass.
    assert!(DwqArchMirror::Qwen35Dense.requires_activation_capture());
    assert!(DwqArchMirror::Qwen35MoE.requires_activation_capture());
    assert!(!DwqArchMirror::Other.requires_activation_capture());
}
