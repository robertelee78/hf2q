//! Speculative-decode primitives (ADR-029).
//!
//! Phase 1 (iter-113, LANDED): pure-CPU n-gram proposer, no model touch.
//! Phase 2 (pending): forward_decode_verify — multi-token verify forward
//!   returning per-position logits + KV-cache rollback.
//! Phase 3 (pending): generate-loop integration (sourdough byte-identity
//!   gate at K=0 enforces production safety until verified).
//!
//! Status: NO production wire-up yet. The proposer module is publicly
//! accessible but no caller exists in `cmd_generate*` until Phase 3.

pub mod dflash;
pub mod ngram_proposer;
pub mod verifier;
