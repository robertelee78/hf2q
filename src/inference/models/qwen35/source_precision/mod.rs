//! Authenticated source-precision Qwen teacher substrate.
//!
//! This module deliberately stops at a retained, structurally verified source
//! snapshot. It does not construct a model, allocate Metal buffers, execute a
//! graph, or mint exact-teacher/sensitivity/allocator authority.

mod header;
mod retained_io;
mod scope;
mod snapshot;
mod types;

pub(crate) use snapshot::{open_verified_qwen_source_snapshot, VerifiedQwenSourceSnapshot};
pub(crate) use types::QwenSourceSnapshotLimits;

#[cfg(test)]
mod tests;
