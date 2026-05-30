//! Speculative-decode primitives (ADR-029).
//!
//! Phase 1 (iter-113, LANDED): pure-CPU n-gram proposer, no model touch.
//! Phase 2 (pending): forward_decode_verify — multi-token verify forward
//!   returning per-position logits + KV-cache rollback.
//! Phase 3 (pending): generate-loop integration (sourdough byte-identity
//!   gate at K=0 enforces production safety until verified).
//!
//! Status: NO production wire-up yet. The proposer module is publicly
//! accessible but no caller exists in `cmd_generate*` until Phase 3.
//!
//! # ADR-040 §6.1.55 (iter-A4-cont-acceptance-telemetry, 2026-05-30)
//!
//! Per-slot spec-decode acceptance-rate telemetry surface for the future
//! empirical inflection-point measurement loop (dossier §1.5 + §6).
//! [`SpecDecodeAcceptanceMetric`] is the structural emission shape: the
//! struct definition + the emission call sites (Qwen35 DFlash target +
//! EAGLE-3 orchestrator) land here today as no-op records — the
//! production telemetry pipeline (`/metrics` schema + scrape) is
//! deferred-on-external-signal per the §6.1.55 dossier framing.

pub mod dflash;
pub mod eagle3;
pub mod eagle3_orchestrator;
pub mod ngram_orchestrator;
pub mod ngram_proposer;
pub mod verifier;

// ──────────────────────────────────────────────────────────────────────────
// ADR-040 §6.1.55 iter-A4-cont-acceptance-telemetry (2026-05-30) —
// per-slot spec-decode acceptance-rate metric.
//
// Structural emission only.  Pure data; no I/O, no allocations beyond
// the record itself.  Callers construct an instance after each
// verification step and route it through [`emit_acceptance_metric`]
// (no-op today — the production wiring is deferred-on-external-signal
// per dossier §6 + §7 + ADR-040 §6.1.55).
//
// **Why land it now**: structural shape pinned at the source-grep level
// (H233 source-greps both the struct + the two emission call sites).
// Operator-runnable telemetry replaces the no-op once the `/metrics`
// schema extension lands at iter-A4-cont-acceptance-telemetry-prod.
// ──────────────────────────────────────────────────────────────────────────

/// **ADR-040 §6.1.55 iter-A4-cont-acceptance-telemetry (2026-05-30)** —
/// per-slot spec-decode acceptance-rate metric.
///
/// Emitted at every speculative-decode verification step from the
/// orchestrator (EAGLE-3 today; DFlash mirrors the same shape).  The
/// metric is the four-field structural shape published in the dossier
/// §1.5 + §6 (accepted vs drafted vs step_count vs slot_id); the
/// production wiring (`/metrics` schema + Prometheus scrape) lands at
/// iter-A4-cont-acceptance-telemetry-prod, gated on operator telemetry
/// infrastructure per dossier §6 + §7.
///
/// # Skip-mode semantics
///
/// In skip mode (no real model loaded) the call sites still construct
/// the record but route it through [`emit_acceptance_metric`] which is
/// a no-op today.  This pins the structural shape at the source-grep
/// level (H233) without engaging any I/O.
///
/// # Cross-references
///
/// - Dossier §1.5 (empirical-inflection-point research source).
/// - Dossier §6 (typed-deferral list — this is `iter-A4-cont-acceptance-telemetry`).
/// - ADR-040 §6.1.55 (closure block — names the full structural
///   bundle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecDecodeAcceptanceMetric {
    /// Multi-seq slot id the metric was measured against.  Pre-A4
    /// (`SerialFifo` / `SlotAware { max_slots: 1 }`) this is always
    /// `SlotId(0)`; with batched spec-decode (post iter-A4-cont-drafter-
    /// dispatcher) it spans `[0, n_seqs)`.
    pub slot_id: crate::serve::multi_seq_kv::SlotId,
    /// Tokens accepted by the verifier in this step (length of the
    /// accepted-tree-walk minus the root).  Matches the existing
    /// `total_accepted_minus_root` accumulator in [`crate::inference::
    /// spec_decode::eagle3_orchestrator::Gemma4Eagle3Orchestrator::
    /// generate`] (`eagle3_orchestrator.rs:1686+`).
    pub accepted_tokens: u32,
    /// Tokens drafted by the speculator in this step (tree length minus
    /// the root).  Matches `total_tree_drafted` at the same call site.
    pub drafted_tokens: u32,
    /// Step counter — the i'th verification step within this generation
    /// loop, starting from 0.  Bookkeeping rides on the orchestrator
    /// (today rebuilt per call; future telemetry pipeline maintains
    /// a per-slot persistent counter).
    pub step_count: u32,
}

impl SpecDecodeAcceptanceMetric {
    /// Construct a metric record.
    ///
    /// Pure constructor; no I/O.
    #[inline]
    pub const fn new(
        slot_id: crate::serve::multi_seq_kv::SlotId,
        accepted_tokens: u32,
        drafted_tokens: u32,
        step_count: u32,
    ) -> Self {
        Self {
            slot_id,
            accepted_tokens,
            drafted_tokens,
            step_count,
        }
    }

    /// Acceptance ratio as `accepted_tokens / drafted_tokens` clamped to
    /// `[0.0, 1.0]`.  Returns `0.0` if `drafted_tokens == 0` (the
    /// degenerate degree-1 tree case — every step accepts the
    /// root-only walk vacuously, but with no draft the ratio is
    /// undefined; caller-friendly default).
    #[inline]
    pub fn acceptance_ratio(self) -> f32 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            let r = self.accepted_tokens as f32 / self.drafted_tokens as f32;
            if r > 1.0 {
                1.0
            } else {
                r
            }
        }
    }
}

/// **ADR-040 §6.1.55 iter-A4-cont-acceptance-telemetry (2026-05-30)** —
/// no-op emission seam for a [`SpecDecodeAcceptanceMetric`].
///
/// Today: no-op (pure record drop; structural emission only).
///
/// At iter-A4-cont-acceptance-telemetry-prod (gated on operator
/// telemetry infrastructure per dossier §6 + §7): routes through the
/// `/metrics` schema for Prometheus scrape.  No allocations on the hot
/// path; the metric is `Copy`.
///
/// Operator-grep cite: `iter-A4-cont-acceptance-telemetry`.
#[inline]
pub fn emit_acceptance_metric(_metric: SpecDecodeAcceptanceMetric) {
    // iter-A4-cont-acceptance-telemetry-prod (deferred): route through
    // /metrics schema.  Today: drop (no-op).  See ADR-040 §6.1.55.
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod adr040_phase_a4_cont_acceptance_telemetry_tests {
    //! ADR-040 §6.1.55 iter-A4-cont-acceptance-telemetry (2026-05-30) —
    //! structural pins for the per-slot spec-decode acceptance metric.

    use super::*;
    use crate::serve::multi_seq_kv::SlotId;

    /// **H233a** — [`SpecDecodeAcceptanceMetric`] carries the four
    /// dossier §1.5 + §6 fields (slot_id, accepted_tokens,
    /// drafted_tokens, step_count) and a stable construction shape.
    #[test]
    fn h233a_spec_decode_acceptance_metric_carries_dossier_shape_2026_05_30() {
        let m = SpecDecodeAcceptanceMetric::new(SlotId(2), 5, 8, 17);
        assert_eq!(m.slot_id, SlotId(2));
        assert_eq!(m.accepted_tokens, 5);
        assert_eq!(m.drafted_tokens, 8);
        assert_eq!(m.step_count, 17);
        // Pure-data witness: derive(Copy) holds.
        let m2 = m;
        assert_eq!(m, m2);
    }

    /// **H233b** — `acceptance_ratio` clamps to `[0.0, 1.0]` and
    /// degenerate `drafted_tokens == 0` returns `0.0` (caller-friendly
    /// default; the production telemetry pipeline filters these rows
    /// from the inflection-point regression).
    #[test]
    fn h233b_acceptance_ratio_clamped_and_degenerate_handled_2026_05_30() {
        // Normal: 5 / 8 = 0.625.
        let m = SpecDecodeAcceptanceMetric::new(SlotId(0), 5, 8, 0);
        assert!((m.acceptance_ratio() - 0.625).abs() < 1e-6);
        // Zero-draft degenerate.
        let m = SpecDecodeAcceptanceMetric::new(SlotId(0), 0, 0, 0);
        assert_eq!(m.acceptance_ratio(), 0.0);
        // Over-accepted (defensive; shouldn't happen but bounded).
        let m = SpecDecodeAcceptanceMetric::new(SlotId(0), 10, 8, 0);
        assert!((m.acceptance_ratio() - 1.0).abs() < 1e-6);
    }

    /// **H233c** — [`emit_acceptance_metric`] is callable as a no-op
    /// today.  The production wiring lands at
    /// iter-A4-cont-acceptance-telemetry-prod per dossier §6 + §7.
    #[test]
    fn h233c_emit_acceptance_metric_is_callable_no_op_2026_05_30() {
        let m = SpecDecodeAcceptanceMetric::new(SlotId(0), 5, 8, 0);
        emit_acceptance_metric(m);
        // No assertion — emission is no-op by design.  Test compiles
        // and runs without panic ⇒ structural shape OK.
    }

    /// **H233d (source-grep pin)** — the emission seam is named at the
    /// two load-bearing call sites (EAGLE-3 orchestrator + Qwen35
    /// DFlash target).
    #[test]
    fn h233d_emission_call_sites_grep_able_2026_05_30() {
        let orchestrator_src =
            include_str!("../../inference/spec_decode/eagle3_orchestrator.rs");
        assert!(
            orchestrator_src.contains("emit_acceptance_metric"),
            "H233d FALSIFIED: eagle3_orchestrator.rs does NOT name \
             `emit_acceptance_metric`.  ADR-040 §6.1.55 \
             iter-A4-cont-acceptance-telemetry requires the EAGLE-3 \
             orchestrator's verification step to construct + emit a \
             SpecDecodeAcceptanceMetric record."
        );
        assert!(
            orchestrator_src.contains("iter-A4-cont-acceptance-telemetry"),
            "H233d FALSIFIED: eagle3_orchestrator.rs does NOT carry the \
             `iter-A4-cont-acceptance-telemetry` cite at the emission \
             site.  Future-iter implementers + operator triage depend \
             on this label being grep-able."
        );
        let dflash_target_src =
            include_str!("../../inference/spec_decode/dflash/qwen35_target.rs");
        assert!(
            dflash_target_src.contains("emit_acceptance_metric"),
            "H233d FALSIFIED: dflash/qwen35_target.rs does NOT name \
             `emit_acceptance_metric`.  ADR-040 §6.1.55 requires the \
             Qwen35 DFlash target's verify path to surface acceptance \
             telemetry the same way EAGLE-3 does."
        );
        assert!(
            dflash_target_src.contains("iter-A4-cont-acceptance-telemetry"),
            "H233d FALSIFIED: dflash/qwen35_target.rs does NOT carry the \
             `iter-A4-cont-acceptance-telemetry` cite at the emission \
             site."
        );
    }
}
