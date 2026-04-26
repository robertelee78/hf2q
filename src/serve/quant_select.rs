//! ADR-005 Phase 3 (lines 901-918): static hardware → quant selection table.
//!
//! Provisional thresholds; refined by measurement before Phase 4 ships. The
//! table lives here in code (per ADR-005:913) and is unit-tested against
//! synthetic [`GpuInfo`] fixtures so boundary cases are deterministic.
//!
//! Decision table (ADR-005:905-911):
//!
//! | Available GPU/unified memory | Quant type |
//! |------------------------------|------------|
//! | ≥ 64 GiB                     | Q8_0       |
//! | 32 – 64 GiB                  | Q6_K       |
//! | 16 – 32 GiB                  | Q4_K_M     |
//! | 8 – 16 GiB                   | Q3_K_M     |
//! | < 8 GiB                      | refuse     |
//!
//! Production callers populate [`GpuInfo`] from
//! [`crate::intelligence::hardware::HardwareProfiler::detect`]
//! (`src/intelligence/hardware.rs:185`), whose `total_memory_bytes`/
//! `available_memory_bytes` fields are bytes-typed exactly as this module
//! expects. Tests construct [`GpuInfo`] directly via [`GpuInfo::from_gib`].
//!
//! Per W50's iter-117 audit (commit 84a6ce3): zero dependencies on other
//! Phase 3 iters — purely a static lookup table + struct + thin
//! `HardwareProfile` adapter.

use anyhow::{anyhow, Result};

/// Canonical GGML quant-type names referenced by the Phase 3 selection
/// table. Matches the string conventions already used throughout hf2q
/// (`src/quantize/apex.rs` `KQuantSpec.name`, `src/backends/gguf.rs`
/// `quant_name_to_ggml_type`, `QuantInfo.ggml_type`).
///
/// Defined locally in this module (rather than reusing
/// [`crate::cli::QuantMethod`]) because `QuantMethod` is the
/// conversion-time CLI surface and does not enumerate fine-grained
/// K-quant variants (Q4_K_M / Q3_K_M etc.) — those are GGML-level
/// type names. ADR-005 Phase 3 selects at GGML granularity, not CLI
/// granularity.
///
/// Variant names intentionally match the GGML wire identifiers
/// (`Q4_K_M`, etc.) rather than upper-camel-case Rust convention; this
/// keeps `as_str()` a bit-identical pass-through and avoids a renaming
/// translation layer at every serialization boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    /// 8-bit per weight, K-block-free legacy quant. Best fidelity in the table.
    Q8_0,
    /// 6.5 bpw K-quant.
    Q6_K,
    /// 4.5 bpw K-quant, "M" mix profile (default for medium-VRAM machines).
    Q4_K_M,
    /// 3.5 bpw K-quant, "M" mix profile (lowest supported in the static table).
    Q3_K_M,
}

impl QuantType {
    /// Canonical GGML name (matches `quant_name_to_ggml_type` strings in
    /// `src/backends/gguf.rs:1038-1057`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Q6_K => "Q6_K",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q3_K_M => "Q3_K_M",
        }
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// GPU / unified-memory descriptor for the Phase 3 selection rule.
///
/// Construction:
/// - **Production**: [`GpuInfo::from_hardware_profile`] adapts
///   [`crate::intelligence::hardware::HardwareProfile`] (Apple Silicon
///   unified memory is reported via `sysinfo::System::total_memory`,
///   `src/intelligence/hardware.rs:191`).
/// - **Tests**: [`GpuInfo::from_gib`] for deterministic boundary fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuInfo {
    /// Total available GPU / unified memory, in bytes. On Apple Silicon
    /// (the primary target per ADR-006) this is the unified-memory pool
    /// shared between CPU and Metal GPU.
    pub memory_bytes: u64,
}

impl GpuInfo {
    /// Construct a fixture from a whole-GiB amount. Used by tests; production
    /// callers should use [`Self::from_hardware_profile`].
    pub fn from_gib(gib: u64) -> Self {
        Self {
            memory_bytes: gib.saturating_mul(1u64 << 30),
        }
    }

    /// Construct from raw bytes. Used by [`Self::from_hardware_profile`] and
    /// any caller that already has a byte count in hand.
    pub fn from_bytes(memory_bytes: u64) -> Self {
        Self { memory_bytes }
    }

    /// Adapter from the hf2q-wide hardware profile. Uses
    /// `available_memory_bytes` rather than `total_memory_bytes` because the
    /// selection rule's intent (per ADR-005:905) is "memory available to host
    /// the model right now," not the machine's nameplate RAM. On a busy host
    /// these can differ by tens of GiB.
    pub fn from_hardware_profile(
        profile: &crate::intelligence::hardware::HardwareProfile,
    ) -> Self {
        Self::from_bytes(profile.available_memory_bytes)
    }

    /// Memory expressed as GiB (1024^3 bytes). `f64` for display only —
    /// the selection rule operates on the integer floor (see
    /// [`select_quant`]) to make boundaries crisp and reproducible.
    pub fn memory_gib_f64(&self) -> f64 {
        self.memory_bytes as f64 / (1u64 << 30) as f64
    }

    /// Integer-floor GiB, the value the selection rule keys off of.
    pub fn memory_gib_floor(&self) -> u64 {
        self.memory_bytes / (1u64 << 30)
    }
}

/// ADR-005 Phase 3 static thresholds, ordered high → low. The first row
/// whose threshold is `≤ available_gib_floor` wins.
///
/// Provisional per ADR-005:903 ("refined by measurement before Phase 4
/// ships"). The *rule shape* — static table, VRAM-indexed, documented —
/// is committed; the numbers are tunable.
const THRESHOLDS_GIB: &[(u64, QuantType)] = &[
    (64, QuantType::Q8_0),
    (32, QuantType::Q6_K),
    (16, QuantType::Q4_K_M),
    (8, QuantType::Q3_K_M),
];

/// Minimum supported configuration. Below this, [`select_quant`] returns
/// `Err` rather than silently picking a lower-fidelity quant. ADR-005:911
/// explicitly mandates the refusal behavior.
pub const MIN_SUPPORTED_GIB: u64 = 8;

/// Select the appropriate quant type for the given hardware.
///
/// Returns `Err` with a clear message naming the minimum supported config
/// when memory is below 8 GiB (ADR-005:911).
///
/// The comparison is on integer-floor GiB so that, e.g., 7.99 GiB is
/// "below 8" (refused) and 8.00 GiB is "exactly 8" (accepted). This makes
/// fixture testing deterministic — see this module's `tests` submodule.
pub fn select_quant(info: &GpuInfo) -> Result<QuantType> {
    let gib_floor = info.memory_gib_floor();
    if gib_floor < MIN_SUPPORTED_GIB {
        return Err(anyhow!(
            "hf2q requires at least {min} GiB of GPU/unified memory; \
             detected {detected} GiB ({bytes} bytes). \
             Minimum supported configuration: {min} GiB → Q3_K_M.",
            min = MIN_SUPPORTED_GIB,
            detected = gib_floor,
            bytes = info.memory_bytes,
        ));
    }
    for &(threshold, quant) in THRESHOLDS_GIB {
        if gib_floor >= threshold {
            return Ok(quant);
        }
    }
    // Unreachable: THRESHOLDS_GIB's lowest entry is 8 GiB and we've already
    // refused < 8 GiB above. If you remove the 8 GiB row, also update
    // MIN_SUPPORTED_GIB so the static contract still holds.
    unreachable!(
        "quant selection fell through table at {gib_floor} GiB — \
         THRESHOLDS_GIB lower bound and MIN_SUPPORTED_GIB are out of sync"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ≥ 64 GiB → Q8_0 ──────────────────────────────────────────────────

    #[test]
    fn select_quant_64_gib_exact_q8() {
        let info = GpuInfo::from_gib(64);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q8_0);
    }

    #[test]
    fn select_quant_just_above_64_gib_q8() {
        // 64 GiB + 1 byte → still Q8_0 (well above threshold)
        let info = GpuInfo::from_bytes((64u64 << 30) + 1);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q8_0);
    }

    #[test]
    fn select_quant_huge_machine_q8() {
        // 128 GiB (e.g., M5 Ultra) → Q8_0
        let info = GpuInfo::from_gib(128);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q8_0);
    }

    // ── 32–64 GiB → Q6_K ─────────────────────────────────────────────────

    #[test]
    fn select_quant_63_gib_q6() {
        let info = GpuInfo::from_gib(63);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q6_K);
    }

    #[test]
    fn select_quant_just_below_64_gib_q6() {
        // (64 GiB − 1 byte) → integer-floor is 63 → Q6_K
        let info = GpuInfo::from_bytes((64u64 << 30) - 1);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q6_K);
    }

    #[test]
    fn select_quant_32_gib_exact_q6() {
        let info = GpuInfo::from_gib(32);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q6_K);
    }

    // ── 16–32 GiB → Q4_K_M ───────────────────────────────────────────────

    #[test]
    fn select_quant_31_gib_q4() {
        let info = GpuInfo::from_gib(31);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q4_K_M);
    }

    #[test]
    fn select_quant_just_below_32_gib_q4() {
        let info = GpuInfo::from_bytes((32u64 << 30) - 1);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q4_K_M);
    }

    #[test]
    fn select_quant_16_gib_exact_q4() {
        let info = GpuInfo::from_gib(16);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q4_K_M);
    }

    // ── 8–16 GiB → Q3_K_M ────────────────────────────────────────────────

    #[test]
    fn select_quant_15_gib_q3() {
        let info = GpuInfo::from_gib(15);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q3_K_M);
    }

    #[test]
    fn select_quant_just_below_16_gib_q3() {
        let info = GpuInfo::from_bytes((16u64 << 30) - 1);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q3_K_M);
    }

    #[test]
    fn select_quant_8_gib_exact_q3() {
        let info = GpuInfo::from_gib(8);
        assert_eq!(select_quant(&info).unwrap(), QuantType::Q3_K_M);
    }

    // ── < 8 GiB → refuse ─────────────────────────────────────────────────

    #[test]
    fn select_quant_7_gib_refuse() {
        let info = GpuInfo::from_gib(7);
        let err = select_quant(&info).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("8 GiB"),
            "error must name the 8 GiB minimum, got: {msg}"
        );
    }

    #[test]
    fn select_quant_just_below_8_gib_refuse() {
        // (8 GiB − 1 byte) → floor is 7 → refused
        let info = GpuInfo::from_bytes((8u64 << 30) - 1);
        assert!(select_quant(&info).is_err());
    }

    #[test]
    fn select_quant_zero_refuse() {
        let info = GpuInfo::from_bytes(0);
        assert!(select_quant(&info).is_err());
    }

    #[test]
    fn select_quant_error_message_names_min() {
        let info = GpuInfo::from_gib(4);
        let err = select_quant(&info).unwrap_err();
        let msg = format!("{err}");
        // Per ADR-005:911, the error must "clearly name the minimum
        // supported config". Verify both the threshold and the fallback
        // quant name appear so an operator can act on the message.
        assert!(msg.contains("8 GiB"), "missing '8 GiB' in: {msg}");
        assert!(msg.contains("Q3_K_M"), "missing 'Q3_K_M' in: {msg}");
        assert!(
            msg.contains("4 GiB"),
            "missing detected size '4 GiB' in: {msg}"
        );
    }

    // ── QuantType surface ────────────────────────────────────────────────

    #[test]
    fn quant_type_as_str_matches_ggml_names() {
        // Names must match `quant_name_to_ggml_type` in
        // src/backends/gguf.rs:1038-1057 so downstream serializers can
        // round-trip without a translation table.
        assert_eq!(QuantType::Q8_0.as_str(), "Q8_0");
        assert_eq!(QuantType::Q6_K.as_str(), "Q6_K");
        assert_eq!(QuantType::Q4_K_M.as_str(), "Q4_K_M");
        assert_eq!(QuantType::Q3_K_M.as_str(), "Q3_K_M");
    }

    #[test]
    fn quant_type_display_matches_as_str() {
        assert_eq!(format!("{}", QuantType::Q4_K_M), "Q4_K_M");
    }

    #[test]
    fn gpu_info_gib_helpers_roundtrip() {
        let info = GpuInfo::from_gib(48);
        assert_eq!(info.memory_gib_floor(), 48);
        assert!((info.memory_gib_f64() - 48.0).abs() < 1e-9);
        assert_eq!(info.memory_bytes, 48u64 << 30);
    }
}
