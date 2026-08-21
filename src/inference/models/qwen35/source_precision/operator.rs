//! Operator-owned assembly for the pinned Qwen3.8 source-teacher gate.
//!
//! This module is the only production bridge from the accepted source manifest
//! and checked-in corpus profile to the opaque source-teacher capabilities. It
//! does not expose caller-authored tensor dispositions, prediction plans, or
//! execution knobs.

mod acceptance;
mod corpus;
mod profile;
mod reference;
mod source;
mod source_manifest;

use serde::{Deserialize, Serialize};

use crate::intelligence::calibration::DatasetSplit;

/// Characterization splits. AcceptanceHoldout is intentionally absent and is
/// reconstructible only through the closed, non-executing evidence verifier.
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

pub(crate) use acceptance::verify_official_qwen38_acceptance_evidence;

pub(crate) use source::{
    preflight_official_qwen38_source_teacher, run_official_qwen38_source_teacher,
    OfficialQwen38SourceTeacherRequestV1,
};

#[cfg(test)]
mod tests;
