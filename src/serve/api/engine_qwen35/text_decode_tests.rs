//! #553: the Qwen slot decode state never splits a UTF-8 character across
//! byte-level tokens. Production `from_prefill_logits`,
//! `commit_ordinary_target_token`, `complete_tick` (applied by
//! `settle_qwen35_decode_lane` to every serial and joint tick) and `finish`
//! run over a weightless model with a byte-level tokenizer.

use std::time::Duration;

use ahash::AHashMap;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

use super::speculation_qualification_tests::loaded_model;
use super::{Qwen35DecodeState, Qwen35LoadedModel, Qwen35TickOutcome};
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
use crate::serve::multi_seq_kv::SlotId;

const A: u32 = 1;
/// Byte-level `â` = E2, the lead byte of U+2082 (`₂` = E2 82 82).
const LEAD: u32 = 2;
/// Byte-level `ĤĤ` = 82 82, the two continuation bytes.
const CONT2: u32 = 3;
/// Byte-level `âĤ` = E2 82, an incomplete sequence.
const PARTIAL: u32 = 4;
/// Byte-level `aâ` = 61 E2: complete text followed by a lead byte.
const A_LEAD: u32 = 5;
const B: u32 = 6;
const EOS: u32 = 63;
const PROMPT: [u32; 3] = [6, 1, 6];

fn byte_level_model() -> Qwen35LoadedModel {
    let pieces = [
        (A, "a"),
        (LEAD, "\u{00E2}"),
        (CONT2, "\u{0124}\u{0124}"),
        (PARTIAL, "\u{00E2}\u{0124}"),
        (A_LEAD, "a\u{00E2}"),
        (B, "b"),
    ];
    let vocab: AHashMap<String, u32> = pieces
        .iter()
        .map(|&(id, piece)| (piece.to_string(), id))
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("byte-level BPE");
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    let mut qwen = loaded_model(QwenSpeculationPolicy::Off);
    assert_eq!(qwen.eos_token_ids, vec![EOS]);
    qwen.tokenizer = tokenizer;
    qwen
}

fn params(max_tokens: usize, stop_strings: &[&str]) -> SamplingParams {
    SamplingParams {
        max_tokens,
        stop_strings: stop_strings.iter().map(|s| s.to_string()).collect(),
        ..SamplingParams::default()
    }
}

fn seed(qwen: &Qwen35LoadedModel, first: u32, params: SamplingParams) -> Qwen35DecodeState {
    let mut logits = vec![0.0_f32; qwen.vocab_size];
    logits[first as usize] = 10.0;
    let state = Qwen35DecodeState::from_prefill_logits(
        qwen,
        PROMPT.to_vec(),
        params,
        None,
        SlotId(0),
        0,
        &logits,
        Duration::ZERO,
        PROMPT.len(),
        None,
    )
    .expect("seed Qwen decode state");
    assert!(!state.finished_at_seed());
    state
}

/// One ordinary tick as the worker settles it.
fn tick(state: &mut Qwen35DecodeState, qwen: &Qwen35LoadedModel, token: u32) -> Qwen35TickOutcome {
    let outcome = state
        .commit_ordinary_target_token(qwen, token, Duration::from_micros(1))
        .expect("ordinary commit");
    state.complete_tick(outcome)
}

#[test]
fn split_character_streams_once_complete_and_eos_flushes_held_bytes() {
    let qwen = byte_level_model();
    // The single-token rendering the stream used to emit.
    assert_eq!(qwen.tokenizer.decode(&[LEAD], false).unwrap(), "\u{FFFD}");
    let mut state = seed(&qwen, A, params(16, &[]));
    let mut stream = vec![state.seed_fragment()];
    for token in [LEAD, CONT2, PARTIAL] {
        let outcome = tick(&mut state, &qwen, token);
        assert!(!outcome.finished);
        stream.push(outcome.fragment);
    }
    let eos = tick(&mut state, &qwen, EOS);
    assert!(eos.finished);
    stream.push(eos.fragment);
    // SSE: nothing until the character completes; EOS flushes the
    // incomplete E2 82 tail as one U+FFFD, like a full-sequence decode.
    assert_eq!(stream, ["a", "", "\u{2082}", "", "\u{FFFD}"]);
    let unary = state.finish(&qwen, None).expect("finish").text;
    assert_eq!(unary, stream.concat());
    assert_eq!(
        unary,
        qwen.tokenizer
            .decode(&[A, LEAD, CONT2, PARTIAL], false)
            .unwrap()
    );
}

#[test]
fn stop_string_discards_held_bytes() {
    let qwen = byte_level_model();
    let mut state = seed(&qwen, B, params(16, &["a"]));
    assert_eq!(state.seed_fragment(), "b");
    let outcome = tick(&mut state, &qwen, A_LEAD);
    assert!(outcome.finished);
    // The stable `a` is surfaced as before; the held E2 after the stop
    // string is dropped, not flushed as U+FFFD.
    assert_eq!(outcome.fragment, "a");
    let result = state.finish(&qwen, None).expect("finish");
    assert_eq!(result.finish_reason, "stop");
    assert_eq!(result.text, "b");
}

#[test]
fn max_tokens_flushes_held_bytes() {
    let qwen = byte_level_model();
    let mut state = seed(&qwen, A, params(2, &[]));
    let outcome = tick(&mut state, &qwen, LEAD);
    assert!(outcome.finished);
    assert_eq!(outcome.fragment, "\u{FFFD}");
    let result = state.finish(&qwen, None).expect("finish");
    assert_eq!(result.finish_reason, "length");
    assert_eq!(result.text, "a\u{FFFD}");
}
