//! Authenticated source-precision Qwen teacher substrate.
//!
//! This module deliberately stops at a retained, structurally verified source
//! snapshot, opaque host-populated shared-Metal upload, and an exact
//! family-owned prepared text graph. It does not execute that graph or mint
//! completion, exact-teacher, sensitivity, performance, or allocator
//! authority.

mod header;
mod retained_io;
mod scope;
mod snapshot;
mod teacher_execution_plan;
mod topology;
mod topology_expected;
mod topology_expected_mtp;
mod types;
mod upload;
mod upload_plan;
mod upload_transform;

pub(crate) use snapshot::{open_verified_qwen_source_snapshot, VerifiedQwenSourceSnapshot};
#[allow(unused_imports)] // consumed by the subsequent completed runner slice
pub(crate) use teacher_execution_plan::{
    preflight_qwen35_source_teacher_execution, Qwen35SourceTeacherRunLimitsV1,
    StructurallyBoundQwen35SourceTeacherWorkV1,
};
#[allow(unused_imports)] // opaque B2a seam consumed by the subsequent Metal-upload slice
pub(crate) use topology::{admit_qwen35_bf16_topology, VerifiedQwen35Bf16TopologyV1};
pub(crate) use types::QwenSourceSnapshotLimits;
#[allow(unused_imports)] // opaque B3 preparation seams consumed by the completed runner
pub(crate) use upload::{
    prepare_qwen35_source_teacher, prepare_qwen35_source_teacher_run_inputs,
    prepare_uploaded_qwen35_source_teacher, PreparedQwen35SourceTeacherRunInputsV1,
    PreparedQwen35SourceTeacherV1, Qwen35SourceTeacherLimitsV1,
    Qwen35SourceTeacherPreparationPolicyV1,
};
#[allow(unused_imports)] // opaque B2b seam consumed by the subsequent teacher runner
pub(crate) use upload::{upload_qwen35_bf16_topology_to_metal, VerifiedQwen35Bf16MetalUploadV1};
#[allow(unused_imports)] // configured by the subsequent teacher runner
pub(crate) use upload_plan::QwenSourceMetalUploadLimits;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod topology_tests;

#[cfg(test)]
mod upload_tests;

#[cfg(test)]
mod teacher_execution_plan_tests;
