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

pub(crate) use reference::{
    compare_official_qwen38_source_reference, OfficialQwen38SourceReferenceRequestV1,
};

pub(crate) use source::{
    preflight_official_qwen38_source_teacher, run_official_qwen38_source_teacher,
    OfficialQwen38SourceTeacherRequestV1,
};

#[cfg(test)]
mod tests;
