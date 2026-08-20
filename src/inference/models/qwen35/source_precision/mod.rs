//! Authenticated source-precision Qwen teacher substrate.
//!
//! This module deliberately stops at a retained, structurally verified source
//! snapshot and opaque, host-populated shared-Metal upload. It does not
//! construct a model, execute a graph, or mint completion, exact-teacher,
//! sensitivity, performance, or allocator authority.

mod header;
mod retained_io;
mod scope;
mod snapshot;
mod topology;
mod topology_expected;
mod topology_expected_mtp;
mod types;
mod upload;
mod upload_plan;
mod upload_transform;

pub(crate) use snapshot::{open_verified_qwen_source_snapshot, VerifiedQwenSourceSnapshot};
#[allow(unused_imports)] // opaque B2a seam consumed by the subsequent Metal-upload slice
pub(crate) use topology::{admit_qwen35_bf16_topology, VerifiedQwen35Bf16TopologyV1};
pub(crate) use types::QwenSourceSnapshotLimits;
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
