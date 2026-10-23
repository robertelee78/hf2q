//! Model-free, declared memory projections for ADR-063 Q5 slice 1.
//!
//! No caller of this module may treat a projection as runtime admission. It
//! observes neither allocations nor OS metrics, and has no serving hook.

mod categories;
mod limits;
mod ownership;

pub use categories::{Category, CategoryKey, Entry, Family, FrozenShape, Inventory, Owner};
pub use limits::{Ceiling, DeclaredBytes, Limits, Projection};
pub use ownership::{Event, HandleId, Ownership, ReservationId, RootId, Transition};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid frozen identity: {0}")]
    Identity(String),
    #[error("invalid or incomplete category: {0}")]
    Category(String),
    #[error("unknown required bound: {0}")]
    Unknown(String),
    #[error("checked memory arithmetic overflow")]
    Overflow,
    #[error("memory limit exceeded: {0}")]
    Limit(&'static str),
    #[error("invalid ownership event: {0}")]
    Ownership(&'static str),
}

/// All three hashes must identify externally frozen bytes. Validation here
/// checks syntax and equality only; it does not attest those bytes or claims.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenIdentity {
    pub artifact_sha256: String,
    pub workload_sha256: String,
    pub transition_plan_sha256: String,
}

impl FrozenIdentity {
    pub fn validate(&self) -> Result<(), Error> {
        for hash in [
            &self.artifact_sha256,
            &self.workload_sha256,
            &self.transition_plan_sha256,
        ] {
            if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err(Error::Identity("expected SHA-256 identity".into()));
            }
        }
        Ok(())
    }
}

/// Receipt scope is fixed by construction, including for fully declared toys.
#[derive(Debug, Serialize)]
pub struct Receipt {
    schema: &'static str,
    evidence_scope: &'static str,
    runtime_qualified: bool,
    identity: FrozenIdentity,
    shape: FrozenShape,
    inventory: Vec<(CategoryKey, Entry)>,
    limits: Limits,
    projection: Projection,
    transition: Transition,
}

impl Receipt {
    pub fn declared(
        inventory: &Inventory,
        expected: &FrozenShape,
        limits: &Limits,
        ownership: &Ownership,
    ) -> Result<Self, Error> {
        inventory.require_shape(expected)?;
        ownership.require_identity(&expected.identity)?;
        limits.require_family(expected.family)?;
        let projection = limits.project(inventory.total()?)?;
        let transition = ownership.snapshot();
        // Traces are an independent cross-check on declared capacity, never
        // additive bytes. Reject a trace exceeding the aggregate declared sum;
        // this does not prove a per-category mapping or attest caller claims.
        if !projection.bytes.covers(transition.peak) {
            return Err(Error::Ownership("trace peak exceeds inventory projection"));
        }
        Ok(Self {
            schema: "hf2q.memory-projection.v1",
            evidence_scope: "caller_declared_model_free_projection_only",
            runtime_qualified: false,
            identity: expected.identity.clone(),
            shape: expected.clone(),
            inventory: inventory.entries(),
            limits: limits.clone(),
            projection,
            transition,
        })
    }
}

#[cfg(test)]
mod tests;
