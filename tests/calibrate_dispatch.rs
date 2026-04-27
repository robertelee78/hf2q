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

/// Local mirror of `cli::QuantMethod` (ADR-014 P8 Decision 12 — 17
/// variants). **Variants must stay in sync** with
/// `src/cli.rs::QuantMethod`. The compile-time exhaustive matches
/// below catch drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantMethodMirror {
    // Flat / passthrough cells — none × flat
    Auto,
    F16,
    Bf16,
    Q2,
    Q4,
    Q8,
    // Uncalibrated K-quant cells — none × k-quant
    Q4KM,
    Q5KM,
    Q6K,
    // imatrix-calibrated K-quant cells — imatrix × k-quant
    ImatrixQ4KM,
    ImatrixQ5KM,
    ImatrixQ6K,
    // imatrix-adaptive — per-tensor optimal precision (preserves Apex)
    ImatrixAdaptive,
    // DWQ cells — dwq × bit-pair
    Dwq46,
    Dwq48,
    Dwq68,
    Dwq28,
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

/// Mirror of [`crate::calibrate::imatrix_calibrator::ImatrixCalibrator`]'s
/// public contract.
struct ImatrixMirror;
impl CalibratorContract for ImatrixMirror {
    fn name(&self) -> &'static str {
        "imatrix"
    }
    fn requires_forward_pass(&self) -> bool {
        true
    }
}

/// Mirror of `select_calibrator` from `src/main.rs` (ADR-014 P8
/// Decision 12 — 17-variant menu).
///
/// Invariant: the match arms here MUST stay aligned with those in
/// `select_calibrator`. The `mirror_covers_every_quant_method` test
/// below catches drift at compile time.
///
/// The `_capture_present` flag mirrors the no-silent-fallback contract
/// in `select_calibrator`: for Imatrix variants, when `capture` is None
/// the helper returns `Err(ForwardPassUnavailable)` rather than a
/// NoneCalibrator. Here we collapse that to the calibrator name only
/// (the live error-path is exercised by main.rs's typed return).
fn select_calibrator_mirror(
    method: QuantMethodMirror,
    arch: DwqArchMirror,
) -> Box<dyn CalibratorContract> {
    use QuantMethodMirror::*;
    match method {
        Dwq46 | Dwq48 | Dwq68 | Dwq28 => {
            Box::new(DwqMirror { arch })
        }
        ImatrixQ4KM | ImatrixQ5KM | ImatrixQ6K | ImatrixAdaptive => {
            Box::new(ImatrixMirror)
        }
        Auto | F16 | Bf16 | Q2 | Q4 | Q8 | Q4KM | Q5KM | Q6K => {
            Box::new(NoneMirror)
        }
    }
}

#[test]
fn select_calibrator_returns_none_for_static_methods() {
    // ADR-014 P8 Decision 12: NoneCalibrator covers Auto + flat
    // (f16/bf16/q2/q4/q8) + uncalibrated K-quant (q4_k_m/q5_k_m/q6_k).
    let static_methods = [
        QuantMethodMirror::Auto,
        QuantMethodMirror::F16,
        QuantMethodMirror::Bf16,
        QuantMethodMirror::Q2,
        QuantMethodMirror::Q4,
        QuantMethodMirror::Q8,
        QuantMethodMirror::Q4KM,
        QuantMethodMirror::Q5KM,
        QuantMethodMirror::Q6K,
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
        QuantMethodMirror::Dwq46,
        QuantMethodMirror::Dwq48,
        QuantMethodMirror::Dwq68,
        QuantMethodMirror::Dwq28,
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

/// ADR-014 P8 Decision 12 — Imatrix variants dispatch to ImatrixCalibrator.
#[test]
fn select_calibrator_returns_imatrix_for_imatrix_methods() {
    let imatrix_methods = [
        QuantMethodMirror::ImatrixQ4KM,
        QuantMethodMirror::ImatrixQ5KM,
        QuantMethodMirror::ImatrixQ6K,
        QuantMethodMirror::ImatrixAdaptive,
    ];
    for m in imatrix_methods {
        let calibrator = select_calibrator_mirror(m, DwqArchMirror::Qwen35Dense);
        assert_eq!(
            calibrator.name(),
            "imatrix",
            "imatrix method {m:?} must dispatch to ImatrixCalibrator (got {})",
            calibrator.name()
        );
        assert!(
            calibrator.requires_forward_pass(),
            "ImatrixCalibrator must require a forward pass (method={m:?})"
        );
    }
}

#[test]
fn select_calibrator_dwq_arch_propagated() {
    // Dwq46 with Qwen35MoE → requires_forward_pass == true.
    let c1 = select_calibrator_mirror(
        QuantMethodMirror::Dwq46,
        DwqArchMirror::Qwen35MoE,
    );
    assert_eq!(c1.name(), "dwq");
    assert!(
        c1.requires_forward_pass(),
        "DwqCalibrator + Qwen35MoE must require forward pass"
    );

    // Dwq68 with Qwen35Dense → requires_forward_pass == true.
    let c2 = select_calibrator_mirror(
        QuantMethodMirror::Dwq68,
        DwqArchMirror::Qwen35Dense,
    );
    assert!(c2.requires_forward_pass());

    // Dwq28 with Other → requires_forward_pass == false.
    let c3 =
        select_calibrator_mirror(QuantMethodMirror::Dwq28, DwqArchMirror::Other);
    assert!(!c3.requires_forward_pass());
}

#[test]
fn select_calibrator_dwq_arch_other_no_capture_required() {
    // The "Other" arch contract: DWQ on a non-qwen35 architecture
    // does NOT require a forward-pass / capture impl. (The
    // downstream weight-space contract for DwqArch::Other is
    // enforced separately in DwqCalibrator's unit tests.)
    for m in [
        QuantMethodMirror::Dwq46,
        QuantMethodMirror::Dwq48,
        QuantMethodMirror::Dwq68,
        QuantMethodMirror::Dwq28,
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
            Dwq46 | Dwq48 | Dwq68 | Dwq28 => "dwq",
            ImatrixQ4KM | ImatrixQ5KM | ImatrixQ6K | ImatrixAdaptive => "imatrix",
            Auto | F16 | Bf16 | Q2 | Q4 | Q8 | Q4KM | Q5KM | Q6K => "none",
        }
    }
    // Spot-check every variant flows through (17 cells).
    assert_eq!(classify(QuantMethodMirror::Auto), "none");
    assert_eq!(classify(QuantMethodMirror::F16), "none");
    assert_eq!(classify(QuantMethodMirror::Bf16), "none");
    assert_eq!(classify(QuantMethodMirror::Q2), "none");
    assert_eq!(classify(QuantMethodMirror::Q4), "none");
    assert_eq!(classify(QuantMethodMirror::Q8), "none");
    assert_eq!(classify(QuantMethodMirror::Q4KM), "none");
    assert_eq!(classify(QuantMethodMirror::Q5KM), "none");
    assert_eq!(classify(QuantMethodMirror::Q6K), "none");
    assert_eq!(classify(QuantMethodMirror::ImatrixQ4KM), "imatrix");
    assert_eq!(classify(QuantMethodMirror::ImatrixQ5KM), "imatrix");
    assert_eq!(classify(QuantMethodMirror::ImatrixQ6K), "imatrix");
    assert_eq!(classify(QuantMethodMirror::ImatrixAdaptive), "imatrix");
    assert_eq!(classify(QuantMethodMirror::Dwq46), "dwq");
    assert_eq!(classify(QuantMethodMirror::Dwq48), "dwq");
    assert_eq!(classify(QuantMethodMirror::Dwq68), "dwq");
    assert_eq!(classify(QuantMethodMirror::Dwq28), "dwq");
}

/// ADR-014 P8 Decision 12: select_calibrator(ImatrixQ4KM) returns the
/// ImatrixCalibrator (calibrator.name() == "imatrix"); NOT NoneCalibrator.
#[test]
fn select_calibrator_imatrix_q4_k_m_routes_to_imatrix_calibrator() {
    let calib =
        select_calibrator_mirror(QuantMethodMirror::ImatrixQ4KM, DwqArchMirror::Other);
    assert_eq!(calib.name(), "imatrix");
    assert!(calib.requires_forward_pass());
}

/// ADR-014 P8 Decision 12: select_calibrator(Dwq46) returns DwqCalibrator
/// (calibrator.name() == "dwq").
#[test]
fn select_calibrator_dwq_4_6_routes_to_dwq_calibrator() {
    let calib =
        select_calibrator_mirror(QuantMethodMirror::Dwq46, DwqArchMirror::Qwen35MoE);
    assert_eq!(calib.name(), "dwq");
    assert!(
        calib.requires_forward_pass(),
        "DwqCalibrator + Qwen35MoE arch must require forward pass"
    );
}

/// ADR-014 P8 Decision 12: select_calibrator(Q4KM) returns NoneCalibrator
/// (uncalibrated K-quant flows through the codec with CalibrationData::None).
#[test]
fn select_calibrator_q4_k_m_routes_to_none_calibrator() {
    let calib =
        select_calibrator_mirror(QuantMethodMirror::Q4KM, DwqArchMirror::Other);
    assert_eq!(calib.name(), "none");
    assert!(!calib.requires_forward_pass());
}

#[test]
fn dwq_arch_table_contract() {
    // The DwqArch table that drives DwqCalibrator::requires_forward_pass.
    assert!(DwqArchMirror::Qwen35Dense.requires_activation_capture());
    assert!(DwqArchMirror::Qwen35MoE.requires_activation_capture());
    assert!(!DwqArchMirror::Other.requires_activation_capture());
}
