//! hf2q library facade — exists to expose `serve::kv_persist` to
//! integration tests under `tests/`. This is an ADDITIVE surface:
//! `main.rs` continues to be the executable entry point and is
//! unaffected (Cargo accepts both an implicit `[lib]` target and the
//! existing `[[bin]]` target on the same package).
//!
//! ## What's exposed
//!
//! Only `serve::kv_persist` (block store + async writer + recovery)
//! consumed by `tests/kv_persist_writer_kill_minus_9.rs`. Other
//! modules (cli, inference, quantize, ...) remain bin-private.
//!
//! ## Why a lib target at all
//!
//! Pre-ADR-017, `hf2q` was bin-only — integration tests couldn't
//! reach production modules. ADR-017 §A.2 ships a `kill -9` mid-write
//! integration test that drives the real `DiskBlockStore` +
//! `AsyncWriterHandle` from a forked child; the test imports the
//! production types directly so the property under test (atomic-
//! rename invariant under SIGKILL) reflects production verbatim. A
//! narrow lib target is the conventional Rust pattern for that.
//!
//! ## How the path is wired
//!
//! `#[path = "serve/kv_persist/mod.rs"]` shortcuts `src/serve/mod.rs`
//! and reads only the kv_persist subtree. This works because
//! kv_persist is self-contained — its source files reference each
//! other via `crate::serve::kv_persist::*`, and from this lib root
//! that path resolves correctly.

#![allow(clippy::missing_safety_doc)]

pub mod serve {
    //! Narrow re-export: only `kv_persist`. The full `serve` module
    //! from `main.rs` stays binary-private.
    #[path = "../serve/kv_persist/mod.rs"]
    pub mod kv_persist;
}
