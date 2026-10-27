//! Fixed reviewed Q1 workload. These are requests to real adapters, not a model
//! emulator. An unimplemented/capability-unsupported action cannot be PASS.
use super::receipt::Family;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Trajectory {
    T0ColdSerial,
    T1Resume64,
    T2ActualCohort,
    T3Chunk64,
    T4ReverseOrder,
    T5GenuineSpeculation,
    T6CancelRestore,
    T7RepeatCold,
    T8RepeatCold,
}
impl Trajectory {
    pub const ALL: [Self; 9] = [
        Self::T0ColdSerial,
        Self::T1Resume64,
        Self::T2ActualCohort,
        Self::T3Chunk64,
        Self::T4ReverseOrder,
        Self::T5GenuineSpeculation,
        Self::T6CancelRestore,
        Self::T7RepeatCold,
        Self::T8RepeatCold,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::T0ColdSerial => "T0",
            Self::T1Resume64 => "T1",
            Self::T2ActualCohort => "T2",
            Self::T3Chunk64 => "T3",
            Self::T4ReverseOrder => "T4",
            Self::T5GenuineSpeculation => "T5",
            Self::T6CancelRestore => "T6",
            Self::T7RepeatCold => "T7",
            Self::T8RepeatCold => "T8",
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub family: Family,
    pub lanes: Vec<Vec<u32>>,
    pub vocab_size: u32,
}
impl Workload {
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0
            || self.lanes.len() != 4
            || self
                .lanes
                .iter()
                .any(|tokens| tokens.len() != 144 || tokens.iter().any(|&t| t >= self.vocab_size))
        {
            return Err("Q1 requires exactly four complete 128+16 token vectors".into());
        }
        if self
            .lanes
            .iter()
            .enumerate()
            .any(|(i, t)| self.lanes[..i].contains(t))
        {
            return Err("Q1 lane identities must have distinct token vectors".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEvidence {
    pub trajectory: Trajectory,
    pub actual_route: String,
    pub positions_per_lane: Vec<usize>,
    pub checked_terminal_calls: u64,
    pub restored_at: Option<usize>,
    pub cancelled_at: Option<usize>,
    pub accepted_draft_rows: Option<u64>,
    pub rejected_draft_rows: Option<u64>,
    pub physical_cohort_rows: Option<Vec<usize>>,
    pub missing_adapter_capabilities: Vec<String>,
}
impl ExecutionEvidence {
    pub fn validate(&self) -> Result<(), String> {
        if self.actual_route.trim().is_empty()
            || self.positions_per_lane != [144; 4]
            || self.checked_terminal_calls == 0
        {
            return Err("Q1 actual execution/terminal position evidence incomplete".into());
        }
        match self.trajectory {
            Trajectory::T1Resume64 if self.restored_at != Some(64) => {
                Err("resume did not restore actual boundary64".into())
            }
            Trajectory::T6CancelRestore
                if self.cancelled_at != Some(64) || self.restored_at != Some(64) =>
            {
                Err("cancel/restore boundary missing".into())
            }
            Trajectory::T5GenuineSpeculation
                if self.accepted_draft_rows.is_none_or(|n| n == 0)
                    || self.rejected_draft_rows.is_none_or(|n| n == 0) =>
            {
                Err("genuine accepted AND rejected proposals required".into())
            }
            Trajectory::T2ActualCohort
                if self
                    .physical_cohort_rows
                    .as_ref()
                    .is_none_or(|rows| !rows.contains(&4)) =>
            {
                Err("actual physical cohort evidence missing; dimensions are insufficient".into())
            }
            _ => Ok(()),
        }
    }
}
