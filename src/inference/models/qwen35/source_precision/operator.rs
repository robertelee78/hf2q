//! Operator-owned assembly for the pinned Qwen3.8 source-teacher gate.
//!
//! This module is the only production bridge from the accepted source manifest
//! and checked-in corpus profile to the opaque source-teacher capabilities. It
//! does not expose caller-authored tensor dispositions, prediction plans, or
//! execution knobs.

mod corpus;
mod profile;
mod reference;
mod source;
mod source_manifest;

use serde::{Deserialize, Serialize};

use crate::intelligence::calibration::DatasetSplit;

/// The characterization splits that may be executed before thresholds exist.
/// AcceptanceHoldout deliberately has no constructor at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OfficialQwen38EvaluationSplitV1 {
    Calibration,
    PolicyValidation,
}

impl OfficialQwen38EvaluationSplitV1 {
    fn dataset_split(self) -> DatasetSplit {
        match self {
            Self::Calibration => DatasetSplit::Calibration,
            Self::PolicyValidation => DatasetSplit::PolicyValidation,
        }
    }
}

pub(crate) use reference::{
    compare_official_qwen38_source_reference, OfficialQwen38SourceReferenceRequestV1,
};

pub(crate) use source::{
    preflight_official_qwen38_source_teacher, run_official_qwen38_source_teacher,
    OfficialQwen38SourceTeacherRequestV1,
};

#[cfg(test)]
mod tests;
