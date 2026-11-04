//! Witness-ON routing adapter: the ONLY place `gemma_actual`, the cohort
//! localizer and the Gemma metadata prepare name `gemma_routing_witness`.
//! Endpoint packets and the `Cohort` localizer reach it; `DriftAll128` never
//! calls it (every call site is behind a non-drift branch), and the drift
//! packet's `routing_witness` stays an opaque JSON value that must be null.
//!
//! The `540a4708` DriftAll128 diagnostic backport (#665 §2.6) keeps
//! `gemma_actual.rs`, `gemma_cohort_localizer.rs` and the cohort export of
//! `gemma_prepare.rs` byte-identical and OMITS this file together with the
//! witness module: it substitutes a witness-free module of the same API whose
//! `budget` refuses any non-null budget, `require_published` refuses, `CHARGE`
//! is 0, `budget_json` is null, `install` returns `Ok(None)` and the row hooks
//! and `finish` are no-ops. Witness-ON endpoint and `Cohort` packets therefore
//! refuse there before model load; only `DriftAll128` runs.
use super::super::{acquisition::Acquisition, gemma_routing_witness as witness, receipt::Identity};
use anyhow::{ensure, Context, Result};
use std::path::Path;
/// Routing bank + metadata + CPU ledger bytes a witness-ON packet charges.
pub(crate) const CHARGE: u64 = witness::PAYLOAD_BYTES + witness::METADATA_BYTES + witness::CPU_LEDGER_BYTES;
/// Parses the packet's opaque budget and admits it exactly (published native).
pub(crate) fn budget(raw: Option<&serde_json::Value>) -> Result<Option<witness::Budget>> {
    let b = raw
        .map(|v| serde_json::from_value::<witness::Budget>(v.clone()))
        .transpose()
        .context("routing witness budget")?;
    witness::validate_budget(b.as_ref())?;
    Ok(b)
}
/// The exact budget a witness-ON packet declares.
pub(crate) fn budget_json() -> serde_json::Value {
    serde_json::json!({"payload_bytes":witness::PAYLOAD_BYTES,"metadata_bytes":witness::METADATA_BYTES,
        "cpu_ledger_bytes":witness::CPU_LEDGER_BYTES})
}
/// Witness-ON packets and routing metadata require the published witness native.
pub(crate) fn require_published(native: &str) -> Result<()> {
    ensure!(
        native == witness::WITNESS_NATIVE_VERSION,
        "exact published routing-witness native build required"
    );
    Ok(())
}
pub(crate) fn install(
    device: &mlx_native::MlxDevice,
    root: &Path,
    identity: &Identity,
    observed: &Acquisition,
    cancelled: bool,
    raw: Option<&serde_json::Value>,
) -> Result<Option<witness::Scope>> {
    let mode = if cancelled { witness::Mode::Cancelled } else { witness::Mode::Ordinary };
    witness::install(device, root, identity, observed, mode, budget(raw)?.as_ref())
}
/// Opens the canonical four-lane batched decode row at `position`.
pub(crate) fn begin_canonical(position: usize) -> Result<()> {
    witness::begin(witness::Namespace::Canonical, (0..4).map(|lane| witness::Row { lane, position }).collect())
}
pub(crate) fn end_checked() -> Result<()> {
    witness::end_checked()
}
pub(crate) fn finish() -> Result<()> {
    witness::finish()
}
