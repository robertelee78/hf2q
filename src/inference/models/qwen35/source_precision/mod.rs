//! Authenticated source-precision Qwen teacher substrate.
//!
//! This module deliberately stops at a retained, structurally verified source
//! snapshot. It does not construct a model, allocate Metal buffers, execute a
//! graph, or mint exact-teacher/sensitivity/allocator authority.

mod header;
mod retained_io;
mod scope;
mod snapshot;
mod topology;
mod topology_expected;
mod topology_expected_mtp;
mod types;

pub(crate) use snapshot::{open_verified_qwen_source_snapshot, VerifiedQwenSourceSnapshot};
#[allow(unused_imports)] // opaque B2a seam consumed by the subsequent Metal-upload slice
pub(crate) use topology::{admit_qwen35_bf16_topology, VerifiedQwen35Bf16TopologyV1};
pub(crate) use types::QwenSourceSnapshotLimits;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod topology_tests;
