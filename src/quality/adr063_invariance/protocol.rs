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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedCall {
    pub lanes: Vec<usize>,
    pub before: Vec<usize>,
    pub input_rows: usize,
    pub after: Vec<usize>,
    pub published: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEvidence {
    pub trajectory: Trajectory,
    pub actual_route: String,
    pub positions_per_lane: Vec<usize>,
    pub checked_terminal_calls: u64,
    pub observed_calls: Vec<ObservedCall>,
    pub restored_at: Option<usize>,
    pub cancelled_at: Option<usize>,
    pub accepted_draft_rows: Option<u64>,
    pub rejected_draft_rows: Option<u64>,
    pub physical_cohort_rows: Option<Vec<usize>>,
    pub missing_adapter_capabilities: Vec<String>,
}
impl ExecutionEvidence {
    /// Called only after the real API returns a checked terminal (or the typed
    /// post-drain cancellation gate fires). Actual cache cursors are supplied.
    pub fn record(
        &mut self,
        lanes: Vec<usize>,
        before: Vec<usize>,
        input_rows: usize,
        after: Vec<usize>,
        published: bool,
    ) {
        self.observed_calls.push(ObservedCall {
            lanes,
            before,
            input_rows,
            after,
            published,
        });
        self.checked_terminal_calls += 1;
    }
    fn validate_axis(&self) -> Result<(), String> {
        let mut expected = Vec::new();
        if self.trajectory == Trajectory::T5GenuineSpeculation {
            return Err("genuine speculation actual call-axis adapter not implemented".into());
        }
        if self.trajectory == Trajectory::T2ActualCohort {
            expected.push(ObservedCall {
                lanes: vec![0, 1, 2, 3],
                before: vec![0; 4],
                input_rows: 128,
                after: vec![128; 4],
                published: true,
            });
            for position in 128..144 {
                expected.push(ObservedCall {
                    lanes: vec![0, 1, 2, 3],
                    before: vec![position; 4],
                    input_rows: 1,
                    after: vec![position + 1; 4],
                    published: true,
                });
            }
        } else {
            let order = if self.trajectory == Trajectory::T4ReverseOrder {
                [3, 2, 1, 0]
            } else {
                [0, 1, 2, 3]
            };
            for lane in order {
                let split = matches!(
                    self.trajectory,
                    Trajectory::T1Resume64 | Trajectory::T3Chunk64 | Trajectory::T6CancelRestore
                );
                if split {
                    expected.push(ObservedCall {
                        lanes: vec![lane],
                        before: vec![0],
                        input_rows: 64,
                        after: vec![64],
                        published: true,
                    });
                    if self.trajectory == Trajectory::T6CancelRestore {
                        expected.push(ObservedCall {
                            lanes: vec![lane],
                            before: vec![64],
                            input_rows: 64,
                            after: vec![64],
                            published: false,
                        });
                    }
                    expected.push(ObservedCall {
                        lanes: vec![lane],
                        before: vec![64],
                        input_rows: 64,
                        after: vec![128],
                        published: true,
                    });
                } else {
                    expected.push(ObservedCall {
                        lanes: vec![lane],
                        before: vec![0],
                        input_rows: 128,
                        after: vec![128],
                        published: true,
                    });
                }
                for position in 128..144 {
                    expected.push(ObservedCall {
                        lanes: vec![lane],
                        before: vec![position],
                        input_rows: 1,
                        after: vec![position + 1],
                        published: true,
                    });
                }
            }
        }
        if self.observed_calls != expected
            || self.checked_terminal_calls != self.observed_calls.len() as u64
        {
            return Err("actual checked call order/chunk/cursor/commit evidence differs from frozen trajectory".into());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.actual_route.trim().is_empty()
            || self.positions_per_lane != [144; 4]
            || self.checked_terminal_calls == 0
        {
            return Err("Q1 actual execution/terminal position evidence incomplete".into());
        }
        self.validate_axis()?;
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

#[cfg(test)]
mod call_axis_controls {
    use super::*;
    #[test]
    fn q1_actual_call_axis_rejects_relabelled_unsplit_and_wrong_order() {
        let mut e = ExecutionEvidence {
            trajectory: Trajectory::T3Chunk64,
            actual_route: "actual-helper fixture".into(),
            positions_per_lane: vec![144; 4],
            checked_terminal_calls: 0,
            observed_calls: vec![],
            restored_at: None,
            cancelled_at: None,
            accepted_draft_rows: None,
            rejected_draft_rows: None,
            physical_cohort_rows: None,
            missing_adapter_capabilities: vec![],
        };
        for lane in 0..4 {
            e.record(vec![lane], vec![0], 64, vec![64], true);
            e.record(vec![lane], vec![64], 64, vec![128], true);
            for position in 128..144 {
                e.record(vec![lane], vec![position], 1, vec![position + 1], true);
            }
        }
        assert_eq!(e.checked_terminal_calls, 72);
        assert!(e.validate().is_ok());
        let mut wrong = e.clone();
        wrong.trajectory = Trajectory::T0ColdSerial;
        assert!(
            wrong.validate().is_err(),
            "split axis cannot be relabelled as cold128"
        );
        let mut wrong = e.clone();
        wrong.observed_calls.swap(0, 18);
        assert!(
            wrong.validate().is_err(),
            "same dimensions/counts cannot hide lane order"
        );
        let mut wrong = e.clone();
        wrong.observed_calls[1].after[0] = 127;
        assert!(
            wrong.validate().is_err(),
            "actual published cursor is mandatory"
        );
    }
}
