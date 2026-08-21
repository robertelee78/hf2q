//! Authenticated source-precision Qwen teacher substrate.
//!
//! Production deliberately stops at a retained, structurally verified source
//! snapshot, opaque host-populated shared-Metal upload, and an exact
//! family-owned prepared text graph. The only authority-bearing execution path
//! is a consuming, one-shot source-teacher transaction; no raw model, cache, or
//! forward session escapes it. It mints exact-teacher completion only after
//! terminal Metal completion and publish-last target sealing, never
//! sensitivity, performance, allocator, selector, or autoquant authority.

mod header;
mod operator;
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

pub(crate) use operator::{
    compare_official_qwen38_acceptance_reference, compare_official_qwen38_source_reference,
    preflight_official_qwen38_acceptance_teacher, preflight_official_qwen38_source_teacher,
    run_official_qwen38_acceptance_teacher, run_official_qwen38_source_teacher,
    OfficialQwen38AcceptanceReferenceRequestV1, OfficialQwen38AcceptanceTeacherRequestV1,
    OfficialQwen38EvaluationSplitV1, OfficialQwen38SourceReferenceRequestV1,
    OfficialQwen38SourceTeacherRequestV1,
};
pub(crate) use snapshot::open_verified_qwen_source_snapshot;
#[cfg(test)]
pub(crate) use snapshot::VerifiedQwenSourceSnapshot;
pub(crate) use teacher_execution_plan::{
    preflight_qwen35_source_teacher_execution, Qwen35SourceTeacherRunLimitsV1,
    StructurallyBoundQwen35SourceTeacherWorkV1,
};
pub(crate) use topology::{admit_qwen35_bf16_topology, VerifiedQwen35Bf16TopologyV1};
#[cfg(test)]
pub(crate) use types::QwenSourceSnapshotLimits;
pub(in crate::inference::models::qwen35) use upload::SourceTeacherCacheAuthorization;
pub(crate) use upload::{
    preflight_qwen35_source_teacher_run_inputs_capacity, prepare_qwen35_source_teacher_run_inputs,
    run_qwen35_source_teacher, Qwen35SourceTeacherCapacityPreflightV1,
    Qwen35SourceTeacherPreparationPolicyV1,
};
pub(crate) use upload_plan::QwenSourceMetalUploadLimits;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod topology_tests;

#[cfg(test)]
mod upload_tests;

#[cfg(test)]
mod teacher_execution_plan_tests;
