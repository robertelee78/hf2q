//! ADR-014 P10 iter-2b — integration smoke tests for
//! `src/quality/ppl_driver.rs::measure_ppl_qwen35_dense`.
//!
//! Five always-on tests exercise the driver's input validation,
//! GGUF error path (missing file + invalid magic), the public
//! `chunk_count` arithmetic, and the `seq_len` override surface.
//!
//! ## Why `#[path]`-include rather than `use hf2q::...`
//!
//! hf2q is a binary crate (no `[lib]` target — confirmed at
//! `Cargo.toml:1-160`). The established pattern for reaching
//! production source files from `tests/*.rs` is `#[path]`-include —
//! see `tests/imatrix_xvalidation.rs:48-52` for the precedent.
//!
//! `ppl_driver.rs` references the `Qwen35Model` public API via
//! `use crate::inference::models::qwen35::...`. That tree
//! (model.rs + ffn.rs + full_attn.rs + delta_net.rs +
//! weight_loader.rs + many submodules — see
//! `src/inference/models/qwen35/mod.rs:28-49`) is too
//! deeply-interconnected to mirror via `#[path]`-include here, and
//! the file fence forbids modifying the qwen35 tree (it's owned by
//! a parallel ADR-013 session). Workaround: provide minimal
//! type-stubs in the `inference::models::qwen35` namespace below.
//! The stubs satisfy the compiler without dragging in the real
//! qwen35 module tree; the always-on tests below never reach the
//! stubbed `Qwen35Model::load_from_gguf` because every test fails
//! the driver at an earlier check (input validation or
//! `GgufFile::open`). P11 will swap this in for a real `[lib]`
//! target so the production qwen35 code path is exercised end-to-end
//! against a real 27B-dense GGUF.

#[path = "../src/quality/perplexity.rs"]
pub mod perplexity;

#[path = "../src/quality/ppl_driver.rs"]
mod ppl_driver;

mod quality {
    //! Aliases the `#[path]`-included `perplexity` mod under
    //! `crate::quality::perplexity` so `ppl_driver.rs`'s
    //! `use crate::quality::perplexity::{...}` line resolves in
    //! this test crate.
    pub use super::perplexity;
}

mod inference {
    //! Type-stubs for `crate::inference::models::qwen35::*`. Never
    //! invoked at runtime by the smoke tests below — every test
    //! exits the driver at input-validation or `GgufFile::open`
    //! before any `Qwen35Model` method is called.

    pub mod models {
        pub mod qwen35 {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub enum Qwen35Variant {
                Dense,
                Moe,
            }

            pub mod forward_cpu {
                /// Mirror of
                /// `src/inference/models/qwen35/forward_cpu.rs::text_positions`
                /// — kept identical (text-convention `[i, i, i, i]`)
                /// so a future change to the production helper trips
                /// a divergence rather than silently desyncing.
                pub fn text_positions(seq_len: u32) -> Vec<[i32; 4]> {
                    (0..seq_len as i32).map(|i| [i, i, i, i]).collect()
                }
            }

            pub mod model {
                use super::Qwen35Variant;
                use anyhow::{anyhow, Result};
                use mlx_native::gguf::GgufFile;

                /// Stub of `Qwen35Config`. Only the fields the
                /// driver reads (`vocab_size`, `max_position_embeddings`,
                /// `variant`) are populated.
                pub struct Qwen35Config {
                    pub vocab_size: u32,
                    pub max_position_embeddings: u32,
                    pub variant: Qwen35Variant,
                }

                /// Stub of `Qwen35Model`. Both methods always error
                /// — the smoke tests never reach them.
                pub struct Qwen35Model {
                    pub cfg: Qwen35Config,
                }

                impl Qwen35Model {
                    pub fn load_from_gguf(_gguf: &GgufFile) -> Result<Self> {
                        Err(anyhow!(
                            "tests/ppl_driver.rs: stubbed Qwen35Model::load_from_gguf \
                             — the production driver lives at \
                             src/quality/ppl_driver.rs and reaches the real \
                             Qwen35Model only when wired through a [lib] target \
                             (deferred to P11)"
                        ))
                    }

                    pub fn forward_cpu(
                        &self,
                        _tokens: &[u32],
                        _positions: &[[i32; 4]],
                    ) -> Result<Vec<f32>> {
                        Err(anyhow!(
                            "tests/ppl_driver.rs: stubbed Qwen35Model::forward_cpu \
                             — see load_from_gguf comment"
                        ))
                    }
                }
            }
        }
    }
}

use ppl_driver::{chunk_count, measure_ppl_qwen35_dense, PplDriverError};

use std::io::Write;
use std::path::Path;

// =====================================================================
// Always-on smoke tests (5)
// =====================================================================

/// Test 1: empty token slice (and length-1 slice) ⇒
/// `PplDriverError::Invalid`. The driver requires at least one
/// prediction (one logits row + one target), which means
/// `tokens.len() >= 2`.
///
/// Verifies the `Invalid` discriminant AND that the message names
/// the offending precondition (`tokens.len()`) so a regression that
/// emits the wrong error reason is caught.
#[test]
fn ppl_driver_returns_invalid_on_empty_token_slice() {
    let any_path = Path::new("/nonexistent/whatever.gguf");

    // Empty slice ⇒ Invalid.
    let result = measure_ppl_qwen35_dense(any_path, &[], None);
    match result {
        Err(PplDriverError::Invalid(msg)) => {
            assert!(
                msg.contains("tokens.len()"),
                "Invalid message must name the offending precondition; got `{msg}`"
            );
        }
        other => panic!("expected Err(Invalid(...)) for empty token slice; got {other:?}"),
    }

    // Length-1 slice ⇒ also Invalid (no prediction possible —
    // `logits[0]` would predict `tokens[1]` which doesn't exist).
    let result = measure_ppl_qwen35_dense(any_path, &[42u32], None);
    match result {
        Err(PplDriverError::Invalid(msg)) => {
            assert!(
                msg.contains("tokens.len()"),
                "Invalid message must name the offending precondition; got `{msg}`"
            );
        }
        other => panic!("expected Err(Invalid(...)) for length-1 token slice; got {other:?}"),
    }
}

/// Test 2: missing GGUF path ⇒ `PplDriverError::Gguf` carrying the
/// caller-supplied path verbatim. The path-round-trip is critical:
/// the markdown table / log line uses it to surface *which* GGUF
/// failed without parsing the source error message.
#[test]
fn ppl_driver_returns_gguf_error_on_missing_path() {
    let missing = Path::new("/nonexistent/path/that/cannot/exist/qwen35.gguf");
    let tokens = [1u32, 2, 3, 4];
    let result = measure_ppl_qwen35_dense(missing, &tokens, Some(2));
    match result {
        Err(PplDriverError::Gguf { path, source }) => {
            assert_eq!(
                path,
                missing.to_path_buf(),
                "Gguf error must round-trip the caller's path; got `{}`",
                path.display()
            );
            // The source lives — useful for the log line — but its
            // exact format is `MlxError`-internal. We don't pin
            // bytes; only that `Display` produces something useful.
            let source_str = format!("{source}");
            assert!(
                !source_str.is_empty(),
                "Gguf error source must produce a non-empty Display"
            );
        }
        other => panic!("expected Err(Gguf {{ ... }}) for missing path; got {other:?}"),
    }
}

/// Test 3: GGUF magic check. Write 8 bytes of zeros to a tempfile
/// (so the file exists and `std::fs::File::open` succeeds, but the
/// GGUF magic `"GGUF"` check fails immediately) ⇒
/// `PplDriverError::Gguf`. Distinct failure surface from test 2:
/// IO succeeds, parse fails. Both must funnel into the same
/// discriminated variant so callers route on the cause without
/// inspecting the source.
#[test]
fn ppl_driver_returns_gguf_error_on_invalid_magic() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    {
        let mut f = std::fs::File::create(tmp.path()).expect("create");
        f.write_all(&[0u8; 8]).expect("write zeros");
        f.sync_all().expect("sync");
    }
    let tokens = [1u32, 2, 3, 4];
    let result = measure_ppl_qwen35_dense(tmp.path(), &tokens, Some(2));
    match result {
        Err(PplDriverError::Gguf { path, source: _ }) => {
            assert_eq!(
                path,
                tmp.path().to_path_buf(),
                "Gguf error must round-trip the caller's path even on parse failure"
            );
        }
        other => panic!("expected Err(Gguf {{ ... }}) for invalid magic; got {other:?}"),
    }
}

/// Test 4: `seq_len = Some(0)` is rejected. The override is invalid
/// because chunking with a zero window is undefined.
///
/// This test exercises the `seq_len` parameter surface (the
/// override branch of the driver). The other obvious surface (a
/// `Some(N)` value smaller than the corpus producing the right
/// chunk count) is tested structurally via the public `chunk_count`
/// helper in test 5 — the spec accepts that simpler test in lieu of
/// adding a `#[cfg(test)]`-gated counter inside ppl_driver.rs (which
/// would add production-surface state for testing-only purposes).
#[test]
fn ppl_driver_seq_len_override_is_respected() {
    let any_path = Path::new("/nonexistent/whatever.gguf");
    let tokens = [10u32, 20, 30, 40];

    // seq_len = Some(0) ⇒ Invalid (named in the message).
    let result = measure_ppl_qwen35_dense(any_path, &tokens, Some(0));
    match result {
        Err(PplDriverError::Invalid(msg)) => {
            assert!(
                msg.contains("seq_len"),
                "Invalid message must name the offending parameter; got `{msg}`"
            );
        }
        other => panic!("expected Err(Invalid(seq_len ...)) for Some(0); got {other:?}"),
    }

    // seq_len = Some(2) on a missing path ⇒ still Gguf (validation
    // passes; open fails). Demonstrates the override is accepted at
    // validation but the file-open failure preempts any chunking.
    let result = measure_ppl_qwen35_dense(any_path, &tokens, Some(2));
    assert!(
        matches!(result, Err(PplDriverError::Gguf { .. })),
        "seq_len = Some(2) with missing path must surface Gguf, not Invalid; got {result:?}"
    );
}

/// Test 5: chunking math is exact. 512 tokens with `seq_len = 128`
/// produces 4 non-overlapping chunks. This is pure arithmetic over
/// the public `chunk_count` helper — no model load — and pins the
/// invariant the production loop relies on.
///
/// Boundary cases (zero inputs, exact-multiple, partial-window
/// rounding, corpus < window, window-of-1) are covered in
/// `src/quality/ppl_driver.rs`'s `#[cfg(test)] mod tests` — kept
/// adjacent to the implementation per the established convention
/// for invariant-preserving helpers (see e.g.
/// `src/quality/perplexity.rs:228-237` for the same pattern).
#[test]
fn ppl_driver_chunk_count_for_512_tokens_with_seq_len_128_is_4() {
    assert_eq!(
        chunk_count(512, 128),
        4,
        "512 tokens / 128-token windows = 4 chunks (exact multiple)"
    );

    // Sanity boundaries that lock the math in two more directions:
    // - 512 / 256 = 2 (exact)
    // - 511 / 128 = 4 (rounds up by 1; 511 = 3*128 + 127)
    // Wait: 511 = 3*128 + 127 ⇒ ceil(511/128) = 4. Correct.
    assert_eq!(chunk_count(512, 256), 2);
    assert_eq!(chunk_count(511, 128), 4);
}
