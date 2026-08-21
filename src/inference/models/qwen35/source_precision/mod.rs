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
pub(crate) use types::QwenSourceSnapshotLimits;
pub(in crate::inference::models::qwen35) use upload::SourceTeacherCacheAuthorization;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod topology_tests;

#[cfg(test)]
mod upload_tests;

#[cfg(test)]
mod teacher_execution_plan_tests;
