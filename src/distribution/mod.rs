//! Secure distribution, installation, and update primitives.
//!
//! ADR-045 deliberately starts with data-only, hostile-input-safe schemas.
//! Parsing a document in this module proves structural validity only; it does
//! not authenticate bytes, establish package ownership, or authorize a
//! filesystem mutation.

// This is intentionally unreachable from production dispatch until the real
// trust root, prepared-version pipeline, and public update authority exist.
#[allow(dead_code)]
pub(crate) mod install_state;
pub mod schema;
#[allow(dead_code)]
mod update_auth;
#[allow(dead_code)]
mod update_transport;
