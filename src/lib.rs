//! hf2q library facade — exists to expose `serve::kv_persist` to
//! integration tests under `tests/`. This is an ADDITIVE surface:
//! `main.rs` continues to be the executable entry point and is
//! unaffected (Cargo accepts both an implicit `[lib]` target and the
//! existing `[[bin]]` target on the same package).
//!
//! ## What's exposed
//!
//! Only the A.1 + A.2 leaves of `serve::kv_persist` (block store +
//! async writer + recovery + format + index) consumed by
//! `tests/kv_persist_writer_kill_minus_9.rs`. Other modules (cli,
//! inference, quantize, full `serve::*`, ...) remain bin-private.
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
//! ## Why spiller is excluded from the lib facade
//!
//! `spiller.rs` (ADR-017 §A.3) implements the `KvSpiller<E>` trait
//! defined in `src/serve/multi_model.rs`. Pulling `multi_model` into
//! the lib transitively requires `intelligence::hardware`,
//! `serve::api::engine`, and a long tail of bin-private modules —
//! defeating the "narrow lib" intent. Instead, the lib enumerates
//! kv_persist's submodules explicitly and OMITS spiller. Spiller is
//! reachable from the bin's `main.rs` (which loads
//! `src/serve/kv_persist/mod.rs` with the full submodule list,
//! including `pub mod spiller;`) and from `--bin hf2q kv_persist`
//! tests. The integration test in `tests/` only needs A.1 + A.2.

#![allow(clippy::missing_safety_doc)]

pub mod serve {
    //! Narrow re-export: only `kv_persist`'s A.1 + A.2 leaves. The
    //! full `serve` module from `main.rs` (multi_model, api, etc.)
    //! stays binary-private. See module docs at the crate root for
    //! the why.
    pub mod kv_persist {
        //! Explicit submodule list — DOES NOT include `spiller`
        //! (A.3 is bin-private; see `src/lib.rs` module docs).
        #[path = "../../serve/kv_persist/block_store.rs"]
        pub mod block_store;
        #[path = "../../serve/kv_persist/format.rs"]
        pub mod format;
        #[path = "../../serve/kv_persist/index.rs"]
        pub mod index;
        #[path = "../../serve/kv_persist/recovery.rs"]
        pub mod recovery;
        #[path = "../../serve/kv_persist/writer.rs"]
        pub mod writer;
    }
}
