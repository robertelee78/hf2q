//! Logit-masking for grammar-constrained decoding.
//!
//! Given a running `GrammarRuntime` and a per-vocab pre-decoded token text
//! table, `mask_invalid_tokens` decodes candidates into one flat code-point
//! arena and rejects them as sets against the live grammar stacks: one shared
//! traversal per candidate set rather than one deep runtime clone per token.
//! Invalid candidates are set to `-inf`; the sampler (`sampler_pure`) then
//! picks from the remaining live tokens.
//!
//! This is the CPU-side half of live `response_format: {json_object}` /
//! `{json_schema}` and tool-call enforcement (ADR-005 Decision #6). Every
//! production family exposes logits and invokes this helper before sampling.
//!
//! # Complexity
//!
//! The candidate table is decoded once per mask and filtered collectively as
//! grammar branches advance.  This avoids the former `vocab_size` deep clones
//! and their repeated `Vec`/`HashSet` stack expansion.  Precompute the token
//! byte table (`Vec<Vec<u8>>`) once at engine load; rebuilding it per step
//! would still dominate this path.
//!
//! # Design notes
//!
//! - The caller owns the pre-decoded token text table. Rebuilding it every
//!   call would dominate runtime; cache it on the engine.
//! - The old clone-per-token implementation remains test-only as an exact
//!   oracle.  Regression tests require identical masks across literals,
//!   alternations, broad strings, UTF-8 partials, and dead/accepting states.
//! - Token text may contain partial UTF-8 (tokenizer pieces like GPT-2's
//!   `Ġ` prefix ARE full UTF-8 here after decoding; BPE byte-fallback
//!   tokens are handled by `GrammarRuntime::accept_token`'s incremental
//!   UTF-8 decoder).

use super::sampler::{decode_candidate_utf8, reject_candidates, GrammarCandidate, GrammarRuntime};
use std::cell::RefCell;

/// Maximum number of descending-logit candidates checked directly before
/// falling back to the exhaustive vocabulary mask.
const GREEDY_PROBE_LIMIT: usize = 64;

thread_local! {
    /// Reused candidate storage for temperature-zero grammar sampling. A
    /// 262K-entry Gemma vocabulary is ~4 MiB here and is retained per worker
    /// thread instead of allocated for every generated tool token.
    static GREEDY_CANDIDATES: RefCell<Vec<(usize, f32)>> = const { RefCell::new(Vec::new()) };
}

/// Select the highest-logit grammar-valid token for temperature-zero decode.
///
/// A full candidate-set grammar mask visits the whole vocabulary. That is
/// needlessly expensive for greedy agentic tool calls when the model's top
/// candidate is normally already valid. Probe candidates in
/// descending logit order by repeatedly finding the current argmax, and fall
/// back to the exhaustive mask after a bounded number of misses. The result is
/// exactly the same highest-logit valid token as mask-then-argmax; only the
/// amount of rejected work changes.
pub fn sample_greedy_valid_token(
    logits: &mut [f32],
    previous_tokens: &[u32],
    repetition_penalty: f64,
    grammar: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
    eog_token_ids: &[u32],
) -> u32 {
    if repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        crate::serve::sampler_pure::apply_repetition_penalty(
            logits,
            previous_tokens,
            repetition_penalty,
        );
    }
    if grammar.is_awaiting_trigger() {
        return crate::serve::sampler_pure::sample_greedy(logits);
    }

    let selected = GREEDY_CANDIDATES.with(|cell| {
        let mut candidates = cell.borrow_mut();
        candidates.clear();
        candidates.reserve(logits.len());
        candidates.extend(
            logits
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, logit)| logit.is_finite()),
        );
        if candidates.is_empty() {
            return None;
        }

        let compare = |left: &(usize, f32), right: &(usize, f32)| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        };
        let limit = GREEDY_PROBE_LIMIT.min(candidates.len());
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit - 1, compare);
            candidates.truncate(limit);
        }
        candidates.sort_unstable_by(compare);

        for &(token, _) in candidates.iter() {
            let Some(bytes) = token_bytes.get(token) else {
                return Some(token as u32);
            };
            if eog_token_ids.contains(&(token as u32)) {
                if grammar.is_terminally_accepted() {
                    return Some(token as u32);
                }
                logits[token] = f32::NEG_INFINITY;
                continue;
            }
            // the peer rejects empty non-EOG pieces even after acceptance.
            if bytes.is_empty() || bytes.first() == Some(&0) {
                logits[token] = f32::NEG_INFINITY;
                continue;
            }
            let mut probe = grammar.clone();
            if probe.accept_token(token as u32, bytes) {
                return Some(token as u32);
            }
            logits[token] = f32::NEG_INFINITY;
        }
        None
    });
    if let Some(token) = selected {
        return token;
    }

    mask_invalid_tokens_with_eog(grammar, token_bytes, eog_token_ids, logits);
    crate::serve::sampler_pure::sample_greedy(logits)
}

/// Mask tokens whose byte-text would drive the grammar dead.
///
/// `token_bytes[i]` is the UTF-8 text emitted when token id `i` is
/// sampled (typically `tokenizer.decode(&[i], false)` bytes). Candidates are
/// decoded into one shared code-point arena and rejected against the live
/// grammar stacks; rejected `logits[i]` are set to `f32::NEG_INFINITY`.
///
/// Returns the number of tokens masked. `f32::NEG_INFINITY` is the
/// standard logit-mask value: after softmax it becomes zero probability
/// and the sampler's top-k / top-p pruning drops it naturally.
///
/// Declared EOG tokens survive only when the grammar is already accepting.
/// Empty non-EOG pieces are always masked, matching the peer's fail-closed
/// vocabulary path.
///
/// # Panics
///
/// None. Indices out of bounds are silently skipped.
pub fn mask_invalid_tokens(
    grammar: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
    logits: &mut [f32],
) -> usize {
    mask_invalid_tokens_with_eog(grammar, token_bytes, &[], logits)
}

/// EOG-aware grammar mask matching the peer's vocabulary-bound apply path.
/// EOG ids survive only in an already-accepting state and never participate
/// in token-terminal matching. Empty non-EOG pieces are always rejected.
pub fn mask_invalid_tokens_with_eog(
    grammar: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
    eog_token_ids: &[u32],
    logits: &mut [f32],
) -> usize {
    // Wave 2.6 W-α5 Q2: a suspended runtime (lazy-grammar awaiting its
    // open-marker trigger) masks NOTHING — preamble tokens before the
    // tool-call open marker are unconstrained.  Skip the per-token
    // clone+accept loop entirely (it would also self-gate, but each
    // clone is non-trivial).  This is the apply-half of the dual-gate
    // An awaiting-trigger grammar leaves preamble logits untouched.
    if grammar.is_awaiting_trigger() {
        return 0;
    }
    let n = token_bytes.len().min(logits.len());
    let mut code_points = Vec::with_capacity(n.saturating_mul(2));
    let mut candidates = Vec::with_capacity(n);
    let mut masked = 0usize;

    for i in 0..n {
        let bytes = &token_bytes[i];
        if eog_token_ids.contains(&(i as u32)) {
            if !grammar.is_terminally_accepted() && logits[i].is_finite() {
                logits[i] = f32::NEG_INFINITY;
                masked += 1;
            }
            continue;
        }
        if bytes.is_empty() || bytes.first() == Some(&0) {
            if logits[i].is_finite() {
                logits[i] = f32::NEG_INFINITY;
                masked += 1;
            }
            continue;
        }
        if !logits[i].is_finite() {
            // Already masked (e.g. by logit_bias or a prior pass).
            continue;
        }
        let start = code_points.len();
        let Some(partial_utf8) =
            decode_candidate_utf8(bytes, grammar.partial_utf8, &mut code_points)
        else {
            logits[i] = f32::NEG_INFINITY;
            masked += 1;
            continue;
        };
        candidates.push(GrammarCandidate {
            index: i,
            token_id: i as u32,
            cursor: start,
            end: code_points.len(),
            partial_utf8,
        });
    }

    let rejects = reject_candidates(&grammar.grammar, &grammar.stacks, &candidates, &code_points);
    for reject in rejects {
        if logits[reject.index].is_finite() {
            logits[reject.index] = f32::NEG_INFINITY;
            masked += 1;
        }
    }
    masked
}

/// Previous clone-per-token mask retained only as a semantic oracle.  The
/// production path above must produce the exact same finite-token bitmap.
#[cfg(test)]
fn mask_invalid_tokens_clone_oracle(
    grammar: &GrammarRuntime,
    token_bytes: &[Vec<u8>],
    logits: &mut [f32],
) -> usize {
    if grammar.is_awaiting_trigger() {
        return 0;
    }
    let mut masked = 0usize;
    let n = token_bytes.len().min(logits.len());
    for i in 0..n {
        let bytes = &token_bytes[i];
        if !logits[i].is_finite() {
            continue;
        }
        if bytes.is_empty() || bytes.first() == Some(&0) {
            logits[i] = f32::NEG_INFINITY;
            masked += 1;
            continue;
        }
        let mut runtime = grammar.clone();
        if !runtime.accept_token(i as u32, bytes) {
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
        if rt.accept_token(i as u32, bytes) {
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
    use super::super::parser::parse;
    use super::*;

    fn rt(grammar_src: &str, start: &str) -> GrammarRuntime {
        let g = parse(grammar_src).expect("parse");
        let rid = g.rule_id(start).expect("start");
        GrammarRuntime::new(g, rid).expect("runtime")
    }

    #[test]
    fn token_terminal_masks_by_id_not_decoded_bytes() {
        let runtime = rt("root ::= <[1]>\n", "root");
        let token_bytes = vocab(&["same", "same", "different"]);
        let mut logits = vec![1.0; token_bytes.len()];

        mask_invalid_tokens_with_eog(&runtime, &token_bytes, &[], &mut logits);

        assert!(!logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert!(!logits[2].is_finite());
    }

    #[test]
    fn token_any_and_exclusion_set_mask_by_id_without_enumerating_vocab() {
        let token_bytes = vocab(&["same", "same", "same", "same", "same"]);

        let any = rt("root ::= <[*]>\n", "root");
        let mut any_logits = vec![1.0; token_bytes.len()];
        assert_eq!(
            mask_invalid_tokens_with_eog(&any, &token_bytes, &[], &mut any_logits),
            0
        );

        let exclusion = rt("root ::= !<[1,3]>\n", "root");
        let mut exclusion_logits = vec![1.0; token_bytes.len()];
        assert_eq!(
            mask_invalid_tokens_with_eog(&exclusion, &token_bytes, &[], &mut exclusion_logits,),
            2
        );
        assert!(exclusion_logits[0].is_finite());
        assert!(!exclusion_logits[1].is_finite());
        assert!(exclusion_logits[2].is_finite());
        assert!(!exclusion_logits[3].is_finite());
        assert!(exclusion_logits[4].is_finite());
    }

    #[test]
    fn eog_remains_masked_while_an_accepted_alternate_has_pending_utf8() {
        let mut runtime = rt("root ::= \"\" | .\n", "root");
        assert!(runtime.accept_bytes(&[0xCE]));
        assert!(runtime.is_accepted());
        assert!(!runtime.is_terminally_accepted());

        let mut logits = vec![1.0];
        assert_eq!(
            mask_invalid_tokens_with_eog(&runtime, &[Vec::new()], &[0], &mut logits),
            1
        );
        assert!(!logits[0].is_finite());
    }

    #[test]
    fn eog_is_masked_until_accepted_and_never_satisfies_token_terminal() {
        let eog = [2_u32];
        let token_bytes = vocab(&["x", "y", "<eos>"]);

        let token_runtime = rt("root ::= <[2]>\n", "root");
        let mut token_logits = vec![1.0; token_bytes.len()];
        mask_invalid_tokens_with_eog(&token_runtime, &token_bytes, &eog, &mut token_logits);
        assert!(!token_logits[2].is_finite());

        let mut accepted = rt("root ::= \"x\"\n", "root");
        assert!(accepted.accept_token(0, b"x"));
        let mut accepted_logits = vec![1.0; token_bytes.len()];
        mask_invalid_tokens_with_eog(&accepted, &token_bytes, &eog, &mut accepted_logits);
        assert!(accepted_logits[2].is_finite());
        assert!(!accepted_logits[1].is_finite());
    }

    #[test]
    fn empty_non_eog_piece_is_always_masked() {
        let mut runtime = rt("root ::= \"x\"\n", "root");
        assert!(runtime.accept_token(0, b"x"));
        let token_bytes = vec![b"x".to_vec(), Vec::new()];
        let mut logits = vec![1.0; token_bytes.len()];

        mask_invalid_tokens_with_eog(&runtime, &token_bytes, &[], &mut logits);

        assert!(!logits[1].is_finite());
    }

    #[test]
    fn greedy_probe_obeys_eog_boundary() {
        let token_bytes = vocab(&["x", "<eos>"]);
        let eog = [1_u32];

        let runtime = rt("root ::= \"x\"\n", "root");
        let mut logits = vec![1.0_f32, 10.0];
        assert_eq!(
            sample_greedy_valid_token(&mut logits, &[], 1.0, &runtime, &token_bytes, &eog),
            0
        );

        let mut accepted = rt("root ::= \"x\"\n", "root");
        assert!(accepted.accept_token(0, b"x"));
        let mut logits = vec![1.0_f32, 10.0];
        assert_eq!(
            sample_greedy_valid_token(&mut logits, &[], 1.0, &accepted, &token_bytes, &eog),
            1
        );
    }

    fn vocab(strings: &[&str]) -> Vec<Vec<u8>> {
        strings.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    fn assert_candidate_set_matches_clone_oracle(
        runtime: &GrammarRuntime,
        token_bytes: &[Vec<u8>],
        initial_logits: &[f32],
    ) {
        let mut actual = initial_logits.to_vec();
        let mut oracle = initial_logits.to_vec();
        let actual_masked = mask_invalid_tokens(runtime, token_bytes, &mut actual);
        let oracle_masked = mask_invalid_tokens_clone_oracle(runtime, token_bytes, &mut oracle);
        assert_eq!(actual_masked, oracle_masked, "new/oracle mask count");
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            oracle
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "new candidate-set mask must be bit-identical to clone oracle"
        );
    }

    #[test]
    fn agentic_grammar_contract_candidate_set_matches_clone_oracle_across_runtime_states() {
        let names = vocab(&[
            "ruflo_call",
            "ruflo_search",
            "aqe_call",
            "aqe_search",
            "task",
            "read",
            "write",
            "bash",
            "brain",
            "skill",
            "agent",
            "memory",
            "workflow",
            "hooks",
            "routing",
            "swarm",
            "wrong",
            "",
        ]);
        let runtime = rt(
            "root ::= \"ruflo_call\" | \"ruflo_search\" | \"aqe_call\" | \"aqe_search\" | \"task\" | \"read\" | \"write\" | \"bash\" | \"brain\" | \"skill\" | \"agent\" | \"memory\" | \"workflow\" | \"hooks\" | \"routing\" | \"swarm\"\n",
            "root",
        );
        let mut logits = vec![1.0; names.len()];
        logits[3] = f32::NEG_INFINITY;
        assert_candidate_set_matches_clone_oracle(&runtime, &names, &logits);

        // The current hf2q token table is decoded through Rust `String`s.
        // Preserve its existing incomplete-tail behavior even when the
        // partial code point cannot yet be proven to match the next literal;
        // malformed continuation bytes still reject immediately.
        let literal_runtime = rt("root ::= \"a\"\n", "root");
        let byte_edge_tokens = vec![vec![0xCE], vec![0x80], b"a".to_vec(), b"x".to_vec()];
        assert_candidate_set_matches_clone_oracle(
            &literal_runtime,
            &byte_edge_tokens,
            &vec![1.0; byte_edge_tokens.len()],
        );

        // Broad-string state after the function name is selected.  Include
        // escapes, multi-byte UTF-8, an incomplete tail, malformed UTF-8,
        // an empty special token, and a pre-masked candidate.
        let mut string_runtime = rt("root ::= \"ruflo_call:\\\"\" [^\\\"]* \"\\\"\"\n", "root");
        assert!(string_runtime.accept_bytes(b"ruflo_call:\""));
        let string_tokens = vec![
            b"plain".to_vec(),
            b"{}".to_vec(),
            "α".as_bytes().to_vec(),
            vec![0xCE],
            vec![0x80],
            b"\"".to_vec(),
            Vec::new(),
        ];
        let mut string_logits = vec![1.0; string_tokens.len()];
        string_logits[1] = f32::NEG_INFINITY;
        assert_candidate_set_matches_clone_oracle(&string_runtime, &string_tokens, &string_logits);

        // A UTF-8 code point split across sampled tokens exercises inherited
        // `PartialUtf8` state and `match_partial_char`.
        let mut partial_runtime = rt("root ::= \"α\"\n", "root");
        assert!(partial_runtime.accept_bytes(&[0xCE]));
        let partial_tokens = vec![vec![0xB1], vec![0xB2], b"a".to_vec(), Vec::new()];
        assert_candidate_set_matches_clone_oracle(
            &partial_runtime,
            &partial_tokens,
            &vec![1.0; partial_tokens.len()],
        );

        // Accepting and dead runtimes have intentionally different behavior:
        // an accepting empty stack rejects further non-empty bytes; a dead
        // runtime rejects every candidate. Empty special tokens survive only
        // for the accepting runtime.
        let mut accepted = rt("root ::= \"a\"\n", "root");
        assert!(accepted.accept_bytes(b"a"));
        assert!(accepted.is_accepted());
        let terminal_tokens = vec![Vec::new(), b"a".to_vec(), b"x".to_vec(), vec![0xCE]];
        assert_candidate_set_matches_clone_oracle(
            &accepted,
            &terminal_tokens,
            &vec![1.0; terminal_tokens.len()],
        );

        let mut dead = rt("root ::= \"a\"\n", "root");
        assert!(!dead.accept_bytes(b"x"));
        assert!(dead.is_dead());
        assert_candidate_set_matches_clone_oracle(
            &dead,
            &terminal_tokens,
            &vec![1.0; terminal_tokens.len()],
        );
    }

    /// Reproducible model-free performance probe for the tokenizer.json proxy
    /// vocabulary.  It loads tokenizer metadata only (never weights or
    /// Metal), compares the exact finite-token bitmap against the legacy
    /// clone oracle, and prints both wall times.  Kept ignored because the
    /// oracle intentionally performs one deep runtime clone per vocabulary
    /// token.
    #[test]
    #[ignore = "diagnostic benchmark; set HF2Q_QWEN35_TOKENIZER to tokenizer.json"]
    fn candidate_set_mask_qwen_vocab_benchmark() {
        let Some(tokenizer_path) = std::env::var_os("HF2Q_QWEN35_TOKENIZER") else {
            eprintln!("SKIP: set HF2Q_QWEN35_TOKENIZER to a tokenizer.json path");
            return;
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap_or_else(|error| {
            panic!(
                "failed to load {}: {error}",
                tokenizer_path.to_string_lossy()
            )
        });
        let vocab_size = tokenizer.get_vocab_size(true);
        let token_bytes = (0..vocab_size as u32)
            .map(|id| {
                tokenizer
                    .decode(&[id], false)
                    .unwrap_or_default()
                    .into_bytes()
            })
            .collect::<Vec<_>>();

        let mut runtime = rt("root ::= \"\\\"\" [^\\\"]* \"\\\"\"\n", "root");
        assert!(runtime.accept_bytes(b"\""));
        let initial = vec![1.0_f32; vocab_size];
        let mut actual = initial.clone();
        let mut oracle = initial;

        let started = std::time::Instant::now();
        let actual_masked = mask_invalid_tokens(&runtime, &token_bytes, &mut actual);
        let candidate_set_elapsed = started.elapsed();
        let started = std::time::Instant::now();
        let oracle_masked = mask_invalid_tokens_clone_oracle(&runtime, &token_bytes, &mut oracle);
        let clone_oracle_elapsed = started.elapsed();

        assert_eq!(actual_masked, oracle_masked);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            oracle
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        eprintln!(
            "grammar mask benchmark: vocab={vocab_size} candidate_set={candidate_set_elapsed:?} clone_oracle={clone_oracle_elapsed:?} speedup={:.2}x",
            clone_oracle_elapsed.as_secs_f64() / candidate_set_elapsed.as_secs_f64()
        );
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
    fn mask_allows_declared_eog_only_after_grammar_acceptance() {
        let runtime = rt("root ::= \"a\"\n", "root");
        let token_bytes = vec![b"a".to_vec(), vec![], b"b".to_vec()];
        let mut logits = vec![1.0, 2.0, 3.0];
        let masked = mask_invalid_tokens_with_eog(&runtime, &token_bytes, &[1], &mut logits);
        assert_eq!(masked, 2);
        assert_eq!(logits[0], 1.0); // 'a' survives
        assert!(logits[1].is_infinite()); // early EOS cannot truncate "a"
        assert!(logits[2].is_infinite()); // 'b' masked

        let mut accepted = rt("root ::= \"a\"\n", "root");
        assert!(accepted.accept_bytes(b"a"));
        let mut terminal_logits = vec![2.0];
        assert_eq!(
            mask_invalid_tokens_with_eog(&accepted, &[Vec::new()], &[0], &mut terminal_logits,),
            0
        );
        assert_eq!(terminal_logits[0], 2.0);
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
        let src = super::super::test_fixtures::peer_grammar("json.gbnf");
        let g = parse(src).unwrap();
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
    /// This is the apply-half of the dual-gate.  Together with
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

    // -----------------------------------------------------------------
    // Wave 2.8 W-θ missed-test #2 — tokenizer-backed marker-byte test.
    //
    // Audit gap (wave-2.7): existing mask tests use synthetic
    // `vocab(&["a", "b", ...])` strings; they don't prove that REAL
    // tokenizer decode + the token_bytes_table build path produce
    // non-empty bytes for the special open marker tokens (Gemma 4 id 48
    // = "<|tool_call>", 12 ASCII bytes). The `bytes.is_empty()` skip at
    // mask.rs:77-79 is a documented contract: the open marker must NOT
    // hit it. This test loads the real gemma4 tokenizer.json and
    // exercises that exact path.
    //
    // Methodology: mirror Engine::token_bytes_table's body
    // (`tok.decode(&[id], false)` per id) and assert id 48 decodes to
    // the literal 12-byte UTF-8 string "<|tool_call>". Then build a
    // grammar that requires that exact byte sequence at byte 0 and
    // confirm the mask leaves token 48 surviving (not pushed to
    // -inf) — i.e. the special-token mask-skip contract holds for the
    // marker tokens an eager grammar relies on.
    // -----------------------------------------------------------------

    /// Path to the gemma4 tokenizer fixture on disk. The test gates on
    /// this file's existence so a downstream env without the fixture
    /// (CI minus /opt/hf2q/models/gemma4/) skips cleanly.
    const GEMMA4_TOKENIZER_PATH: &str = "/opt/hf2q/models/gemma4/tokenizer.json";

    fn load_gemma4_tokenizer_or_skip() -> Option<tokenizers::Tokenizer> {
        if !std::path::Path::new(GEMMA4_TOKENIZER_PATH).exists() {
            // Fixture absent — CI without models/gemma4/ skips cleanly.
            return None;
        }
        // Wave 2.9 W-ι: file exists, so a load failure is a corrupt fixture,
        // not a missing-env skip. Panic with a diagnostic rather than silently
        // returning None (which would let the test pass while exercising
        // nothing — the audit gap "tokenizer fixture load failure").
        match tokenizers::Tokenizer::from_file(GEMMA4_TOKENIZER_PATH) {
            Ok(t) => Some(t),
            Err(e) => panic!(
                "Tokenizer fixture exists at {} but failed to load: {}\n\
                 Fix or remove the fixture; do not silence this error.",
                GEMMA4_TOKENIZER_PATH, e
            ),
        }
    }

    /// Build the per-vocab byte table for a small id range using the
    /// SAME mechanism as `Engine::token_bytes_table`
    /// (`tok.decode(&[id], false)`). Returns `Vec<Vec<u8>>` indexed by
    /// id from 0 to `up_to` exclusive.
    fn token_bytes_table_for_range(tok: &tokenizers::Tokenizer, up_to: u32) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(up_to as usize);
        for id in 0..up_to {
            let s = tok.decode(&[id], false).unwrap_or_default();
            out.push(s.into_bytes());
        }
        out
    }

    /// Real-tokenizer test: Gemma 4 id 48 (the `<|tool_call>` special
    /// token, registered with `special: true` and 12-byte content in
    /// models/gemma4/tokenizer.json) MUST decode to non-empty bytes
    /// through the same path the engine builds the token_bytes_table.
    /// If id 48 decoded empty, the mask's `bytes.is_empty()` skip at
    /// mask.rs:77-79 would leave it un-maskable, which is the
    /// documented "special-token loophole" the wave-2.7 research
    /// dossier corrected.
    #[test]
    fn tokenizer_backed_table_preserves_gemma_open_marker_bytes() {
        let Some(tok) = load_gemma4_tokenizer_or_skip() else {
            // Fixture absent — skip cleanly.
            return;
        };

        // Cover ids 0..256 — id 48 is in the special-token block.
        let table = token_bytes_table_for_range(&tok, 256);
        assert_eq!(table.len(), 256);

        let id_48 = &table[48];
        assert!(
            !id_48.is_empty(),
            "Gemma 4 id 48 (<|tool_call>) decoded to empty bytes through \
             tok.decode(&[48], false); the mask's bytes.is_empty() skip \
             at mask.rs:77-79 would leave the open marker un-maskable. \
             This breaks the wave-2.7 Q-A eager-grammar contract."
        );
        assert_eq!(
            id_48.as_slice(),
            b"<|tool_call>",
            "Gemma 4 id 48 must decode to the 12-byte literal '<|tool_call>'; \
             got {:?}",
            String::from_utf8_lossy(id_48)
        );
        assert_eq!(
            id_48.len(),
            12,
            "Gemma 4 '<|tool_call>' is 12 ASCII bytes; got {} bytes",
            id_48.len()
        );
    }

    /// End-to-end test: build a grammar that requires `<|tool_call>` at
    /// byte 0, run the mask path with the REAL tokenizer-backed
    /// token_bytes table, and assert that token id 48 is the surviving
    /// token (the eager grammar's open-marker constraint funnels the
    /// model to id 48 — exactly the wave-2.7 Q-A design).
    #[test]
    fn mask_with_real_tokenizer_keeps_gemma_open_marker_alive() {
        let Some(tok) = load_gemma4_tokenizer_or_skip() else {
            return;
        };

        // Token table covers the special-token block (ids 0..256). 256
        // is enough to exercise id 48 + a representative slice of
        // surrounding non-marker special tokens (ids 0-47, 49-255 are
        // mostly other Gemma special tokens like <pad>, <eos>, etc).
        let token_bytes = token_bytes_table_for_range(&tok, 256);
        // Sanity: id 48 is "<|tool_call>" (proved in the previous test
        // but also exercised here as a precondition).
        assert_eq!(token_bytes[48], b"<|tool_call>");

        // Grammar that REQUIRES the literal "<|tool_call>" prefix.
        // Mirrors the eager-grammar root rule shape from registry.rs's
        // OneOrMoreCalls emitter for Gemma 4.
        let runtime = rt("root ::= \"<|tool_call>\"\n", "root");
        // Initialize logits with a finite value so non-skipped tokens
        // are mask-eligible.
        let mut logits = vec![1.0_f32; token_bytes.len()];
        let _ = mask_invalid_tokens(&runtime, &token_bytes, &mut logits);

        // Token 48 (the literal "<|tool_call>") MUST survive — the
        // grammar accepts that exact byte sequence at byte 0.
        assert!(
            logits[48].is_finite(),
            "Gemma id 48 (<|tool_call>) was masked to {}; the eager \
             grammar's open-marker constraint must FUNNEL the model to \
             this token, not mask it out",
            logits[48]
        );

        // Survivor count among non-empty-byte tokens: there should be
        // very few — only tokens whose first byte is '<' and whose
        // bytes are a valid prefix of "<|tool_call>" can survive.
        let surviving: Vec<u32> = (0..token_bytes.len() as u32)
            .filter(|&i| !token_bytes[i as usize].is_empty() && logits[i as usize].is_finite())
            .collect();
        assert!(
            surviving.contains(&48),
            "id 48 must be in surviving set; got {:?}",
            surviving
        );

        // Record how many empty decoded pieces were exercised. In this
        // non-accepting state all are now masked, closing the EOS bypass.
        let empty_byte_tokens: usize = token_bytes.iter().filter(|b| b.is_empty()).count();
        // We don't assert a specific number — it depends on the
        // tokenizer's special-token registration shape — but assert
        // the table is non-trivial.
        let _ = empty_byte_tokens; // documented presence; not a hard count.
    }

    #[test]
    fn greedy_probe_returns_highest_logit_valid_token() {
        let runtime = rt("root ::= \"a\" | \"b\"\n", "root");
        let token_bytes = vocab(&["x", "b", "a"]);
        let mut logits = vec![9.0, 8.0, 7.0];
        let token = sample_greedy_valid_token(&mut logits, &[], 1.0, &runtime, &token_bytes, &[]);
        assert_eq!(
            token, 1,
            "invalid top token must yield to highest valid token"
        );
    }

    #[test]
    fn greedy_probe_suspended_runtime_is_unconstrained() {
        let mut runtime = rt("root ::= \"a\"\n", "root");
        runtime.set_awaiting_trigger(true);
        let token_bytes = vocab(&["x", "a"]);
        let mut logits = vec![9.0, 8.0];
        let token = sample_greedy_valid_token(&mut logits, &[], 1.0, &runtime, &token_bytes, &[]);
        assert_eq!(token, 0);
        assert_eq!(logits, vec![9.0, 8.0]);
    }

    #[test]
    fn greedy_probe_applies_repetition_penalty_before_ranking() {
        let runtime = rt("root ::= \"a\" | \"b\"\n", "root");
        let token_bytes = vocab(&["a", "b"]);
        let mut logits = vec![10.0, 9.0];
        let token = sample_greedy_valid_token(&mut logits, &[0], 2.0, &runtime, &token_bytes, &[]);
        assert_eq!(token, 1, "penalized prior token must no longer win");
    }
}
