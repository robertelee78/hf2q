use std::fs::File;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::intelligence::calibration::TeacherPredictionPointKind;

pub const EXACT_TEACHER_TARGET_SCHEMA_VERSION: u32 = 1;
pub const EXACT_TEACHER_GREEDY_TOKEN_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherTargetArtifactLimits {
    pub max_vocabulary_size: usize,
    pub max_prediction_rows: usize,
    pub max_target_bytes: u64,
    pub top_k: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherTopKLogit {
    pub token_id: u32,
    pub logit_f32_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherTargetRowReceipt {
    pub point_ordinal: usize,
    pub stable_id: String,
    pub point_kind: TeacherPredictionPointKind,
    pub prefix_token_count: usize,
    pub prefix_token_ids_sha256: String,
    pub vocabulary_size: usize,
    pub payload_offset: u64,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub argmax_token_id: u32,
    pub top_k: Vec<TeacherTopKLogit>,
    pub logsumexp_f64_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherGreedyTrajectoryReceipt {
    pub stable_id: String,
    pub prompt_token_ids_sha256: String,
    pub token_ids: Vec<u32>,
    pub token_ids_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactTeacherTargetReceipt {
    pub schema_version: u32,
    pub semantics: String,
    pub prediction_plan_sha256: String,
    pub limits: TeacherTargetArtifactLimits,
    pub vocabulary_size: usize,
    pub prediction_point_count: usize,
    pub generation_prompt_count: usize,
    pub target_artifact: ArtifactEvidence,
    pub rows: Vec<TeacherTargetRowReceipt>,
    pub greedy_trajectories: Vec<TeacherGreedyTrajectoryReceipt>,
    pub receipt_sha256: String,
}

/// A fully reread and byte-verified retained target file with no execution claim.
///
/// Its constructor is private to the structural writer. This type proves
/// framing and arithmetic only; it cannot be converted into allocator input.
pub(crate) struct StructurallyVerifiedTeacherTargetArtifact {
    pub(super) receipt: ExactTeacherTargetReceipt,
    pub(super) _file: File,
    pub(super) path: PathBuf,
}

impl StructurallyVerifiedTeacherTargetArtifact {
    pub(crate) fn receipt(&self) -> &ExactTeacherTargetReceipt {
        &self.receipt
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub(in crate::intelligence::exact_teacher) fn retained_file_mut(&mut self) -> &mut File {
        &mut self._file
    }
}

#[cfg(test)]
pub(crate) struct TeacherTargetLogitRequest<'a> {
    pub point: &'a crate::intelligence::calibration::TeacherPredictionPointReceipt,
    pub prefix_token_ids: &'a [u32],
}

#[cfg(test)]
pub(crate) struct TeacherGreedyRequest<'a> {
    pub prompt: &'a crate::intelligence::calibration::TeacherGreedyPromptReceipt,
    pub prompt_token_ids: &'a [u32],
}

#[derive(Debug, Error)]
pub enum ExactTeacherTargetError {
    #[error("invalid exact-teacher target contract: {0}")]
    Invalid(String),
    #[error("exact-teacher target I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("exact-teacher target serialization failed: {0}")]
    Serialization(String),
    #[error("exact-teacher logit producer failed: {0}")]
    Producer(String),
}

impl ExactTeacherTargetError {
    pub(super) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            source,
        }
    }
}
