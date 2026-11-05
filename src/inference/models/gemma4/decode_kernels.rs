//! ADR-063 Q4-GEMMA-DECODE switch (`HF2Q_GEMMA_DECODE_KERNELS`, design §4.8,
//! adoption frozen by M0 §7.5).
//!
//! Read once at Gemma 4 model load:
//! - unset or `0`: today's decode path, unchanged (the kill switch);
//! - `1`: the adopted set, F2 only — at batched decode widths `n <= 4` the
//!   routed-expert Q4_K gate_up `mv_id` and `moe_swiglu_batch` pair of the
//!   `[N,hidden]` batched body becomes one expert-grouped gate_up matvec with
//!   the GEGLU epilogue (`quantized_matmul_id_ggml_mv_grouped_geglu`, R0 = 4);
//! - anything else: a startup error.
//!
//! F2's reference is the stock pair. The batched body pins that gate_up to
//! the forced `mv_id` entry, and under the process (environment) policy the
//! Q4_K `mv_id` kernel is fixed: `HF2Q_MM_ID_ROUTING_THRESHOLD` only steers
//! the unforced route, the `nr2` flags cover Q6_K/Q8_0 only, and
//! `dense_decode_mv_ext` only gates the expert-grouped plan, which the
//! environment never enables. No flag changes the reference, so `1` refuses
//! nothing. Other weight types, widths above 4, other families and the serial
//! decode path keep the stock route.
use std::sync::OnceLock;

use anyhow::{bail, Result};
use mlx_native::GgmlType;

pub(crate) const ENV: &str = "HF2Q_GEMMA_DECODE_KERNELS";

/// Largest batched decode width that takes F2.
pub(crate) const F2_MAX_WIDTH: usize = 4;
/// Names of the adopted levers, for the banner and report.
pub(crate) const ADOPTED_LEVERS: &[&str] = &["f2_gate_up_geglu"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Switch {
    /// Unset or `0`: stock decode kernels.
    Stock,
    /// `1`: the adopted routes.
    Adopted,
}

pub(crate) fn parse_switch(raw: Option<&str>) -> Result<Switch> {
    match raw {
        None | Some("0") => Ok(Switch::Stock),
        Some("1") => Ok(Switch::Adopted),
        Some(other) => bail!("{ENV} must be 0, 1 or unset, got {other:?}"),
    }
}

/// The adopted decode set (F2 only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GemmaDecodeKernels {
    _adopted: (),
}

impl GemmaDecodeKernels {
    pub(crate) const fn adopted() -> Self {
        Self { _adopted: () }
    }

    /// Whether the routed-expert gate_up of a width-`n` batched decode step
    /// with a `ty` gate_up stack takes F2 (Q4_K and `1 <= n <= 4`).
    pub(crate) fn f2_gate_up_geglu(&self, ty: GgmlType, n: usize) -> bool {
        ty == GgmlType::Q4_K && (1..=F2_MAX_WIDTH).contains(&n)
    }
}

/// Resolve the switch.
pub(crate) fn resolve(raw: Option<&str>) -> Result<Option<GemmaDecodeKernels>> {
    Ok(match parse_switch(raw)? {
        Switch::Stock => None,
        Switch::Adopted => Some(GemmaDecodeKernels::adopted()),
    })
}

static RUNTIME_REPORT: OnceLock<serde_json::Value> = OnceLock::new();

/// Read the switch from the process environment (once per Gemma model load),
/// log the banner and publish the runtime-control report.
pub(crate) fn resolve_at_load() -> Result<Option<GemmaDecodeKernels>> {
    let raw = std::env::var(ENV).ok();
    let kernels = resolve(raw.as_deref())?;
    tracing::info!("{}", banner(kernels.as_ref()));
    let _ = RUNTIME_REPORT.set(report(kernels.as_ref()));
    Ok(kernels)
}

/// The runtime-control report of the loaded Gemma model; `null` before a
/// Gemma 4 load.
pub(crate) fn runtime_report() -> serde_json::Value {
    RUNTIME_REPORT.get().cloned().unwrap_or(serde_json::Value::Null)
}

/// One-line load banner: switch, adopted levers and their width bound.
pub(crate) fn banner(kernels: Option<&GemmaDecodeKernels>) -> String {
    match kernels {
        None => format!("gemma decode kernels: {ENV}=0 (stock decode kernels)"),
        Some(_) => format!(
            "gemma decode kernels: {ENV}=1 levers=[{}] max_width={F2_MAX_WIDTH} \
             reference=mv_id+moe_swiglu_batch (no refused flags)",
            ADOPTED_LEVERS.join(",")
        ),
    }
}

/// Machine-readable runtime-control record.
pub(crate) fn report(kernels: Option<&GemmaDecodeKernels>) -> serde_json::Value {
    match kernels {
        None => serde_json::json!({"switch": "0"}),
        Some(_) => serde_json::json!({
            "switch": "1",
            "levers": ADOPTED_LEVERS,
            "max_width": F2_MAX_WIDTH,
            "reference": "mv_id+moe_swiglu_batch",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_kernels_switch_parsing() {
        assert_eq!(parse_switch(None).unwrap(), Switch::Stock);
        assert_eq!(parse_switch(Some("0")).unwrap(), Switch::Stock);
        assert_eq!(parse_switch(Some("1")).unwrap(), Switch::Adopted);
        for bad in ["", "2", "true", "on", " 1", "01"] {
            let e = parse_switch(Some(bad)).unwrap_err().to_string();
            assert!(e.contains(ENV), "{bad:?}: {e}");
        }
    }

    #[test]
    fn decode_kernels_resolve() {
        assert!(resolve(None).unwrap().is_none());
        assert!(resolve(Some("0")).unwrap().is_none());
        assert!(resolve(Some("x")).is_err());
        assert_eq!(resolve(Some("1")).unwrap(), Some(GemmaDecodeKernels::adopted()));
    }

    #[test]
    fn decode_kernels_f2_only_q4k_up_to_width_4() {
        let k = GemmaDecodeKernels::adopted();
        for n in 1..=4 {
            assert!(k.f2_gate_up_geglu(GgmlType::Q4_K, n), "n={n}");
        }
        for n in [0, 5, 8, 16] {
            assert!(!k.f2_gate_up_geglu(GgmlType::Q4_K, n), "n={n}");
        }
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            for n in 1..=4 {
                assert!(!k.f2_gate_up_geglu(ty, n), "{ty:?} n={n}");
            }
        }
    }

    #[test]
    fn decode_kernels_banner_and_report() {
        assert_eq!(banner(None), "gemma decode kernels: HF2Q_GEMMA_DECODE_KERNELS=0 (stock decode kernels)");
        let k = GemmaDecodeKernels::adopted();
        assert_eq!(
            banner(Some(&k)),
            "gemma decode kernels: HF2Q_GEMMA_DECODE_KERNELS=1 levers=[f2_gate_up_geglu] max_width=4 \
             reference=mv_id+moe_swiglu_batch (no refused flags)"
        );
        assert_eq!(report(None), serde_json::json!({"switch": "0"}));
        assert_eq!(
            report(Some(&k)),
            serde_json::json!({
                "switch": "1",
                "levers": ["f2_gate_up_geglu"],
                "max_width": 4,
                "reference": "mv_id+moe_swiglu_batch",
            })
        );
    }
}
