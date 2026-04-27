//! Logit-masking for grammar-constrained decoding.
//!
//! Given a running `GrammarRuntime` and a per-vocab pre-decoded token text
//! table, `mask_invalid_tokens` walks every candidate token, clones the
//! grammar, feeds the token's bytes through it, and sets the corresponding
//! logit to `-inf` if the grammar would die on that token. The sampler
//! (sampler_pure) then picks from the remaining live tokens.
//!
//! This is the CPU-side half of `response_format: {json_object}` /
//! `{json_schema}` enforcement (ADR-005 Decision #6). The GPU-side half is
//! the `forward_decode` refactor that exposes logits to the caller so this
//! helper can be invoked per decode step — that refactor is deferred to an
//! iter that has a live model for byte-identical validation.
//!
//! # Complexity
//!
//! `mask_invalid_tokens` is `O(vocab_size × avg_token_bytes × avg_stack_depth)`
//! per decode step. For vocab=262k and a shallow JSON grammar this is
//! ~1-5ms/token on modern CPU — acceptable for correctness-first. When
//! performance matters, precompute a per-token byte table (`Vec<Vec<u8>>`)
//! once at engine load, not per call.
//!
//! # Design notes
//!
//! - The caller owns the pre-decoded token text table. Rebuilding it every
//!   call would dominate runtime; cache it on the engine.
//! - We clone the `GrammarRuntime` per-token (cheap: `Grammar` is already
//!   `Clone`, stacks are small `Vec`s). An alternative using an explicit
//!   rollback API would avoid the clones but complicates the state machine
//!   — the clone approach is simpler and sufficient.
//! - Token text may contain partial UTF-8 (tokenizer pieces like GPT-2's
//!   `Ġ` prefix ARE full UTF-8 here after decoding; BPE byte-fallback
//!   tokens are handled by `GrammarRuntime::accept_bytes`'s incremental
//!   UTF-8 decoder).

use super::sampler::GrammarRuntime;

/// Mask tokens whose byte-text would drive the grammar dead.
///
/// `token_bytes[i]` is the UTF-8 text emitted when token id `i` is
/// sampled (typically `tokenizer.decode(&[i], false)` bytes). For every
/// `i`, a clone of `grammar` consumes `token_bytes[i]`; if the clone
/// dies (no surviving stacks), `logits[i]` is set to `f32::NEG_INFINITY`.
///
/// Returns the number of tokens masked. `f32::NEG_INFINITY` is the
/// standard logit-mask value: after softmax it becomes zero probability
/// and the sampler's top-k / top-p pruning drops it naturally.
///
/// Tokens whose `token_bytes` entry is empty (e.g. special `<|endoftext|>`
/// tokens without a printable form) are **NOT** masked — they're left at
/// their original logit so the sampler can pick them. The caller is
/// responsible for stop-sequence / EOS handling; the grammar doesn't
/// govern them.
///
/// # Panics
///
/// None. Indices out of bounds are silently skipped.
pub fn mask_invalid_tokens(
    grammar: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
    logits: &mut [f32],
) -> usize {
    // Wave 2.6 W-α5 Q2: a suspended runtime (lazy-grammar awaiting its
    // open-marker trigger) masks NOTHING — preamble tokens before the
    // tool-call open marker are unconstrained.  Skip the per-token
    // clone+accept loop entirely (it would also self-gate, but each
    // clone is non-trivial).  This is the apply-half of the dual-gate
    // pattern from /opt/llama.cpp/src/llama-grammar.cpp:1339-1344
    // (`if (grammar.awaiting_trigger) return;`).
    if grammar.is_awaiting_trigger() {
        return 0;
    }
    let mut masked = 0usize;
    let n = token_bytes.len().min(logits.len());
    for i in 0..n {
        let bytes = &token_bytes[i];
        if bytes.is_empty() {
            // Special/unprintable token — don't mask.
            continue;
        }
        if !logits[i].is_finite() {
            // Already masked (e.g. by logit_bias or a prior pass).
            continue;
        }
        let mut rt = grammar.clone();
        let alive = rt.accept_bytes(bytes);
        if !alive {
            logits[i] = f32::NEG_INFINITY;
            masked += 1;
        }
    }
    masked
}

/// Same as `mask_invalid_tokens` but returns the list of token ids that
/// survive (finite logit). Useful for tests + metrics reporting. Does not
/// mutate `logits`.
#[cfg(test)]
pub fn surviving_token_ids(
    grammar: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
    logits: &[f32],
) -> Vec<u32> {
    let mut out = Vec::new();
    let n = token_bytes.len().min(logits.len());
    for i in 0..n {
        let bytes = &token_bytes[i];
        if bytes.is_empty() || !logits[i].is_finite() {
            // Special or pre-masked tokens count as "alive" for the caller.
            if logits[i].is_finite() {
                out.push(i as u32);
            }
            continue;
        }
        let mut rt = grammar.clone();
        if rt.accept_bytes(bytes) {
            out.push(i as u32);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::parse;

    fn rt(grammar_src: &str, start: &str) -> GrammarRuntime {
        let g = parse(grammar_src).expect("parse");
        let rid = g.rule_id(start).expect("start");
        GrammarRuntime::new(g, rid).expect("runtime")
    }

    fn vocab(strings: &[&str]) -> Vec<Vec<u8>> {
        strings.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    #[test]
    fn mask_rejects_tokens_that_dont_match_literal() {
        // Grammar accepts only "abc". Vocab: ["a", "b", "c", "x", "Z"].
        // From the initial state (empty prefix), only "a" is a valid first
        // character. All others die immediately.
        let runtime = rt("root ::= \"abc\"\n", "root");
        let token_bytes = vocab(&["a", "b", "c", "x", "Z"]);
        let mut logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(masked, 4, "only 'a' should survive from {:?}", logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
        assert!(logits[4].is_infinite() && logits[4] < 0.0);
    }

    #[test]
    fn mask_respects_char_class_range() {
        // Grammar accepts a single digit [0-9]. Vocab includes digits,
        // letters, and a multi-char token. All non-digits should die.
        let runtime = rt("root ::= [0-9]\n", "root");
        let token_bytes = vocab(&["0", "5", "9", "a", "ZZ"]);
        let mut logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(masked, 2);
        assert_eq!(logits[0], 1.0); // '0'
        assert_eq!(logits[1], 1.0); // '5'
        assert_eq!(logits[2], 1.0); // '9'
        assert!(logits[3].is_infinite());
        assert!(logits[4].is_infinite()); // 'ZZ' starts with Z — invalid first char
    }

    #[test]
    fn mask_accepts_multi_byte_utf8_token() {
        // Greek alpha (U+03B1, UTF-8 0xCE 0xB1). Token vocab has a
        // two-byte UTF-8 slice — must be accepted by accept_bytes.
        let runtime = rt("root ::= \"α\"\n", "root");
        let token_bytes = vec!["α".as_bytes().to_vec(), "β".as_bytes().to_vec()];
        let mut logits = vec![1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(masked, 1);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite());
    }

    #[test]
    fn mask_skips_empty_token_strings() {
        // Empty-string tokens (special tokens) are left unmasked regardless
        // of grammar state.
        let runtime = rt("root ::= \"a\"\n", "root");
        let token_bytes = vec![b"a".to_vec(), vec![], b"b".to_vec()];
        let mut logits = vec![1.0, 2.0, 3.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(masked, 1); // only 'b' masked
        assert_eq!(logits[0], 1.0); // 'a' survives
        assert_eq!(logits[1], 2.0); // empty token untouched
        assert!(logits[2].is_infinite()); // 'b' masked
    }

    #[test]
    fn mask_ignores_already_negative_infinity_tokens() {
        // A token pre-masked by another pass (e.g. logit_bias) should not
        // be re-evaluated; its logit stays at -inf.
        let runtime = rt("root ::= \"a\" | \"b\"\n", "root");
        let token_bytes = vocab(&["a", "b", "c"]);
        let mut logits = vec![1.0, f32::NEG_INFINITY, 3.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        // 'a' survives; 'b' already masked; 'c' gets masked.
        assert_eq!(masked, 1); // only 'c' is newly masked
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite());
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn mask_is_idempotent_after_running_twice() {
        // Running the mask twice produces the same result: already-masked
        // tokens are skipped (finite-check) and survivors don't flip.
        let runtime = rt("root ::= \"a\" | \"b\"\n", "root");
        let token_bytes = vocab(&["a", "b", "c", "d"]);
        let mut logits = vec![1.0; 4];
        let m1 = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        let m2 = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(m1, 2); // 'c', 'd'
        assert_eq!(m2, 0); // nothing new to mask
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 1.0);
        assert!(logits[2].is_infinite());
        assert!(logits[3].is_infinite());
    }

    #[test]
    fn mask_after_partial_decode_narrows_survivors() {
        // Grammar: "ab". Before any char: only 'a' valid. After accepting
        // 'a': only 'b' valid. Simulates the decode-step progression.
        let mut runtime = rt("root ::= \"ab\"\n", "root");
        let token_bytes = vocab(&["a", "b", "c"]);

        // Step 1 — before any chars accepted.
        let mut logits = vec![1.0, 1.0, 1.0];
        mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite());
        assert!(logits[2].is_infinite());

        // Caller samples 'a' → advance runtime.
        assert!(runtime.accept_char('a' as u32));

        // Step 2 — 'b' becomes valid, others die.
        let mut logits = vec![1.0, 1.0, 1.0];
        mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert!(logits[0].is_infinite());
        assert_eq!(logits[1], 1.0);
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn mask_with_json_grammar_accepts_opening_brace() {
        // Use the canonical json.gbnf fixture. From root=object, the only
        // valid first char is '{' — every token starting with any other
        // char must be masked.
        let src = std::fs::read_to_string("/opt/llama.cpp/grammars/json.gbnf")
            .expect("json.gbnf fixture");
        let g = parse(&src).unwrap();
        let rid = g.rule_id("root").unwrap();
        let runtime = GrammarRuntime::new(g, rid).unwrap();
        let token_bytes = vocab(&["{", "}", "[", "\"", "a", "1"]);
        let mut logits = vec![1.0; 6];
        let _ = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        // '{' survives (root → object → '{' ...)
        assert_eq!(logits[0], 1.0, "'{{' must survive");
        // '}' is invalid at root — must be masked.
        assert!(logits[1].is_infinite(), "'}}' must be masked");
        // '[' is not a top-level object start — masked by `root ::= object`.
        assert!(logits[2].is_infinite(), "'[' must be masked");
        // '"' is not a top-level object start either.
        assert!(logits[3].is_infinite(), "'\"' must be masked");
        // 'a' is invalid.
        assert!(logits[4].is_infinite());
        // '1' is invalid.
        assert!(logits[5].is_infinite());
    }

    #[test]
    fn surviving_token_ids_helper_matches_mask_counts() {
        let runtime = rt("root ::= \"abc\"\n", "root");
        let token_bytes = vocab(&["a", "b", "c", "x"]);
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let survivors = surviving_token_ids(&runtime, &token_bytes, &logits);
        assert_eq!(survivors, vec![0u32]); // only 'a'
    }

    #[test]
    fn mask_does_not_exceed_logits_length() {
        // Defensive: token_bytes can be longer than logits (caller uses a
        // larger vocab cache). mask_invalid_tokens should stop at
        // logits.len().
        let runtime = rt("root ::= \"a\"\n", "root");
        let token_bytes = vocab(&["a", "b", "c", "d", "e"]);
        let mut logits = vec![1.0, 1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(masked, 2);
        assert_eq!(logits.len(), 3);
    }

    /// Wave 2.6 W-α5 Q2 — mask self-gates on awaiting_trigger.
    ///
    /// When the runtime is suspended (lazy grammar awaiting its
    /// trigger), `mask_invalid_tokens` MUST mask zero tokens.  Every
    /// preamble token (e.g. arbitrary text before the tool-call open
    /// marker) stays at its original logit so the model is free to emit
    /// any text up to the trigger.
    ///
    /// This is the apply-half of the dual-gate from
    /// /opt/llama.cpp/src/llama-grammar.cpp:1339-1344
    /// (`if (grammar.awaiting_trigger) return;`).  Together with
    /// `accept_bytes` self-gating (sampler.rs::runtime_accept_noops_when_awaiting_trigger),
    /// this proves the wave-2.5 audit divergence A1 cannot recur:
    /// there is no split-state window where mask says "off" but
    /// advance says "on" because BOTH gate the same boolean.
    #[test]
    fn runtime_apply_noops_when_awaiting_trigger() {
        // Restrictive grammar: only "a" is valid.  Without the gate,
        // 3 of 4 tokens would be masked.
        let mut runtime = rt("root ::= \"a\"\n", "root");
        runtime.set_awaiting_trigger(true);
        let token_bytes = vocab(&["a", "b", "c", "x"]);
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(
            masked, 0,
            "suspended runtime MUST mask zero tokens (preamble freedom)"
        );
        // All logits MUST be unchanged — the model is unconstrained.
        for (i, &l) in logits.iter().enumerate() {
            assert_eq!(l, 1.0, "logit {i} must be unchanged while awaiting trigger");
        }
    }

    /// Wave 2.6 W-α5 Q2 — mask resumes restrictive enforcement after
    /// `trigger()` is called.  Companion to
    /// `runtime_apply_noops_when_awaiting_trigger`: proves the gate is
    /// the ONLY thing suppressing the mask, and that the underlying
    /// grammar is intact.
    #[test]
    fn runtime_apply_active_after_trigger() {
        let mut runtime = rt("root ::= \"a\"\n", "root");
        runtime.set_awaiting_trigger(true);
        runtime.trigger();
        assert!(!runtime.is_awaiting_trigger());

        let token_bytes = vocab(&["a", "b", "c", "x"]);
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(
            masked, 3,
            "post-trigger runtime masks the 3 invalid tokens (only 'a' survives)"
        );
        assert!(logits[0].is_finite(), "'a' survives");
        assert!(logits[1].is_infinite(), "'b' masked");
        assert!(logits[2].is_infinite(), "'c' masked");
        assert!(logits[3].is_infinite(), "'x' masked");
    }

    /// Wave 2.6 W-α5 Q2 — `GrammarKind::ResponseFormat` runtimes (the
    /// default) MUST never await a trigger.  This guards the audit
    /// divergence "A1 / response_format regression" — any code path
    /// that constructs a runtime without explicitly opting into
    /// `set_awaiting_trigger(true)` must enforce eagerly from token 0.
    ///
    /// The test is a property check: a freshly-constructed runtime
    /// reports `is_awaiting_trigger() == false`, and the mask fires
    /// normally without any explicit `trigger()` call.
    #[test]
    fn runtime_response_format_never_awaits() {
        // Default-constructed runtime — no `set_awaiting_trigger` call.
        // This mirrors the engine's GrammarKind::ResponseFormat path.
        let runtime = rt("root ::= \"a\"\n", "root");
        assert!(
            !runtime.is_awaiting_trigger(),
            "default (ResponseFormat-equivalent) runtime MUST NOT await trigger"
        );

        // Mask fires immediately, no trigger needed.
        let token_bytes = vocab(&["a", "b"]);
        let mut logits = vec![1.0, 1.0];
        let masked = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);
        assert_eq!(
            masked, 1,
            "ResponseFormat-kind runtime enforces from token 0 with no \
             trigger flip required"
        );
    }
}
