//! Source capability is not execution, extent coverage, or backend acceptance.
//! This table routes to actual adapters; absent adapters remain incomplete.
use super::{protocol::Trajectory, receipt::{Disposition, Family, Identity}};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Adapter {
    QwenOrdinary,
    DeepseekOrdinary,
    GemmaOrdinary,
    GemmaPackedCohort,
    GemmaOwningWorkerRecovery,
    QwenOwningWorkerRecovery,
    QwenCoherentSpeculation,
    GemmaNgramTransaction,
    SourceRefusal,
    Missing,
}
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub family: Family,
    pub trajectory: Trajectory,
    pub adapter: Adapter,
    pub scope: &'static str,
    pub gap: &'static str,
}
pub fn cell(family: Family, trajectory: Trajectory) -> Cell {
    use Adapter::*;
    use Family::*;
    use Trajectory::*;
    let (adapter, scope, gap) = match (family, trajectory) {
        (Qwen38 | Qwen36, T2ActualCohort) => (SourceRefusal,
            "ordinary decode_batch iterates independent request handles",
            "no shared four-agent ordinary target graph; temporal MTP rows are not agents"),
        (Qwen38 | Qwen36, T5GenuineSpeculation) => (QwenCoherentSpeculation,
            "engine_qwen35::q1_speculation real K3 accepted/rejected transaction",
            "separate coherent ordinary reference; full workload frozen before comparison; Qwen36 actual execution pending"),
        (Qwen38 | Qwen36, T6CancelRestore) => (QwenOwningWorkerRecovery,
            "engine::q1_qwen_cancel real Pending64 then dropped receiver after checked Ready",
            "current continuation captures rows64..143, not all144 logits; Qwen36 actual execution pending"),
        (Qwen38 | Qwen36, _) => (QwenOrdinary,
            "qwen_runner actual forward_gpu and full active endpoint cache",
            "generic recipe/profile and four-lane actual evidence must be bound before qualification"),
        (Deepseek4, T5GenuineSpeculation) => (Missing,
            "selected artifact has no implemented genuine proposal adapter",
            "missing adapter is not proof of source-unsupported capability"),
        (Deepseek4, T6CancelRestore) => (Missing,
            "legacy deliberate native commit-gate rejection is not owning-client cancellation",
            "real session/reply/reservation recovery adapter and explicit prior native64 anchor precondition pending"),
        (Deepseek4, T2ActualCohort) => (DeepseekOrdinary,
            "cooperative packed FFN prefill; decode is independent row graphs",
            "operator-specific physical membership required; no whole-body B4 claim"),
        (Deepseek4, _) => (DeepseekOrdinary,
            "deepseek_runner actual ordinary verifier forward and same-owner T1 restore",
            "authored full active cache/endpoint and same-owner exact64 T1; real execution/resource review pending; owning T6 separate"),
        (Gemma4, T6CancelRestore) => (GemmaOwningWorkerRecovery,
            "actual Gemma Pending64 then checked Ready128/client_closed with production anchor recovery",
            "authored source only; independently exact active64 restore, full endpoint and reservation release proof pending"),
        (Gemma4, T2ActualCohort) => (GemmaPackedCohort,
            "actual four-sequence packed prefill; production batched body and fused-or-batched head decode",
            "authored source adapter; operator-specific row owners, actual execution/native witness pending"),
        (Gemma4, T0ColdSerial | T1Resume64 | T3Chunk64 | T4ReverseOrder | T7RepeatCold | T8RepeatCold) => (GemmaOrdinary,
            "gemma_actual production slot-aware Hybrid body; same scalar head for all actual rows",
            "authored diagnostic; actual execution, exact checkpoint and full cache coverage pending"),
        (Gemma4, T5GenuineSpeculation) => (GemmaNgramTransaction,
            "gemma_ngram global OptionC injected K3 transaction vs same-route ordinary arm",
            "not the slot-aware serving body"),
    };
    Cell { family, trajectory, adapter, scope, gap }
}
pub fn catalogue() -> Vec<Cell> {
    [Family::Qwen38, Family::Deepseek4, Family::Qwen36, Family::Gemma4]
        .into_iter().flat_map(|family| Trajectory::ALL.into_iter()
            .map(move |trajectory| cell(family, trajectory))).collect()
}
/// Called before the generic runner primes a cache or issues a forward.
/// Real T5/T6 require the separate owning-worker APIs, not this generic graph.
pub fn ordinary_preflight(family: Family, trajectory: Trajectory) -> Result<(), String> {
    match cell(family, trajectory).adapter {
        Adapter::QwenOrdinary | Adapter::DeepseekOrdinary | Adapter::GemmaOrdinary => Ok(()),
        Adapter::SourceRefusal => Err(format!("SourceCapabilityRefusal: {}", cell(family, trajectory).gap)),
        Adapter::QwenCoherentSpeculation | Adapter::QwenOwningWorkerRecovery =>
            Err(format!("SeparateActualAdapterRequired: {}", cell(family, trajectory).scope)),
        Adapter::GemmaOwningWorkerRecovery => Err("SeparateActualAdapterRequired: Gemma owning cancellation endpoint filter".into()),
        Adapter::GemmaPackedCohort => Err("SeparateActualAdapterRequired: Gemma packed cohort endpoint filter".into()),
        Adapter::GemmaNgramTransaction => Err(format!("SeparateActualAdapterRequired: {}", cell(family, trajectory).scope)),
        Adapter::Missing => Err(format!("LocalizationIncomplete: {}", cell(family, trajectory).gap)),
    }
}

/// A normalization receipt records separate axes. It never turns a diagnostic
/// backend finding into a whole-matrix or independent task-quality verdict.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub identity: Identity,
    pub trajectory: Trajectory,
    pub adapter: Adapter,
    pub execution_receipt_sha256: String,
    pub trace_sha256: Vec<String>,
    pub reference: Reference,
    pub actual_logical_lanes: u32,
    pub actual_logit_rows_per_lane: u32,
    pub full_active_extent_captured: bool,
    pub backend_finding: Disposition,
    pub missing_capabilities: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Reference {
    ColdT0 { dataset_sha256: String },
    SeparateCoherentOrdinary { dataset_sha256: String, receipt_sha256: String,
        additional_logical_positions: u32 },
}
fn sha(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Receipt {
    pub fn validate(&self) -> Result<(), String> {
        self.identity.validate()?;
        let selected = cell(self.identity.family, self.trajectory);
        if self.adapter != selected.adapter || matches!(self.adapter, Adapter::Missing | Adapter::SourceRefusal)
            || !sha(&self.execution_receipt_sha256) || self.trace_sha256.is_empty()
            || self.trace_sha256.iter().any(|s| !sha(s))
            || self.actual_logical_lanes == 0 || self.actual_logical_lanes > 4
            || self.actual_logit_rows_per_lane == 0 || self.actual_logit_rows_per_lane > 144 {
            return Err("invalid actual adapter/bound execution scope".into());
        }
        let dataset = match &self.reference {
            Reference::ColdT0 { dataset_sha256 } => {
                if self.trajectory == Trajectory::T5GenuineSpeculation {
                    return Err("T5 requires separately charged coherent ordinary reference".into());
                }
                dataset_sha256
            }
            Reference::SeparateCoherentOrdinary { dataset_sha256, receipt_sha256,
                additional_logical_positions } => {
                if self.trajectory != Trajectory::T5GenuineSpeculation
                    || !sha(receipt_sha256) || *additional_logical_positions != 576 {
                    return Err("invalid separate T5 reference/resource scope".into());
                }
                dataset_sha256
            }
        };
        if dataset != &self.identity.token_vectors_sha256 {
            return Err("reference uses a different frozen dataset".into());
        }
        Ok(())
    }
    /// Descriptive completeness only. Never independent oracle/task acceptance.
    pub fn full_matrix_cell_extent(&self) -> Result<bool, String> {
        self.validate()?;
        Ok(self.actual_logical_lanes == 4 && self.actual_logit_rows_per_lane == 144
            && self.full_active_extent_captured && self.missing_capabilities.is_empty())
    }
}

#[cfg(test)]
#[path = "matrix_controls.rs"]
mod controls;
