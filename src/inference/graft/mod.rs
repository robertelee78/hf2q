//! KV-cache grafts — context-space steering artifacts (ADR-059).
//!
//! A graft is a small bank of per-layer key/value tensors spliced into the
//! attention cache as fabricated history at invariant positions `0..N`
//! (phantom-kv's technique). No weight edits; unloading leaves the base
//! byte-identical; the model's own attention does the steering — nothing
//! is projected out of any activation and no per-token hook runs.
//!
//! Container: GGUF with `graft.*` metadata, mirroring the GLP
//! reader-conformance discipline (ADR-053). Fail-closed everywhere:
//! unknown mode, unknown or not-yet-implemented splice site, malformed
//! shapes, hash mismatch, or missing RoPE declaration are fatal — the
//! reader never guesses and never silently mis-splices.
//!
//! Universality split (ADR-059 "Splice-site reality"): the *container* is
//! family-independent data; the *splice site* is per-architecture and
//! enumerated (`full_attn_kv` ships; `window_tail_kv`, `compressed_kv`,
//! `recurrent_state` are staged sites that this reader rejects until
//! they ship); the *artifact* is always checkpoint-bound (cache K is
//! post-RoPE, so a bank is bound to one exact model revision and RoPE
//! configuration).
//!
//! Scope of this module: loading + conformance. Binding (layer/head/RoPE
//! validation against a model path), the cache splice, graft-region
//! tagging, and cache-identity integration live with the family serve
//! paths per the site contracts.

pub mod bind;
pub mod compatibility;
pub mod reader;

pub use bind::{graft_params_hash, BoundGraft};
pub use compatibility::{validate_graft_bank_for_model, validate_graft_for_model, GraftModelShape};
pub use reader::{GraftBank, GraftError, GraftHookPoint, GraftKind, GraftLayerKv, GraftMode};
