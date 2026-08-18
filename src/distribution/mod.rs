//! Secure distribution, installation, and update primitives.
//!
//! ADR-045 deliberately starts with data-only, hostile-input-safe schemas.
//! Parsing a document in this module proves structural validity only; it does
//! not authenticate bytes, establish package ownership, or authorize a
//! filesystem mutation.

pub mod schema;
