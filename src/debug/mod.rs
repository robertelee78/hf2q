//! Investigation/debug scaffolding for hf2q.
//!
//! Category-4 environment variables (see `docs/shipping-contract.md`) are
//! parsed through `investigation_env::InvestigationEnv` — one parse path
//! per process. Hot-path code reads from `INVESTIGATION_ENV.<field>`
//! instead of calling `std::env::var` directly. This is enforced by
//! convention (and will be loud at startup once S-4 lands the warning +
//! unsafe-ack gate).

pub mod dumps;
pub mod investigation_env;

pub use investigation_env::INVESTIGATION_ENV;
