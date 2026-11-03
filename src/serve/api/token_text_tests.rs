use std::collections::HashSet;

use ahash::AHashMap;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

use super::{byte_level_byte, token_raw_bytes, TokenTextDecoder};
use crate::serve::api::grammar::mask::{mask_invalid_tokens_with_eog, sample_greedy_valid_token};
use crate::serve::api::grammar::parser::parse;
use crate::serve::api::grammar::sampler::GrammarRuntime;

const REPLACEMENT: &str = "\u{FFFD}";

/// A Gemma4-shaped tokenizer: byte-fallback BPE over `<0x00>`..`<0xFF>` plus
/// a few text pieces, decoded by the production Gemma4 decoder chain.
fn gemma_like_tokenizer() -> Tokenizer {
    let mut pieces = vec!["<unk>".to_string()];
    pieces.extend((0..=255u8).map(|byte| format!("<0x{byte:02X}>")));
    for piece in [
        "a",
        "2",
        "3",
        "(1-",
        "\u{2581}a",
        REPLACEMENT,
        "call",
        ":",
        "write",
        "_",
        "file",
        "{",
        "}",
        "content",
        "path",
        "<|\"|>",
    ] {
        pieces.push(piece.to_string());
    }
    let vocab: AHashMap<String, u32> = pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| (piece.clone(), id as u32))
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .byte_fallback(true)
        .unk_token("<unk>".to_string())
        .build()
        .expect("byte-fallback BPE");
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_decoder(Some(
        crate::inference::models::gemma4::tokenizer::gemma4_decoder().expect("gemma4 decoder"),
    ));
    tokenizer
}

fn id(tokenizer: &Tokenizer, piece: &str) -> u32 {
    tokenizer
        .token_to_id(piece)
        .unwrap_or_else(|| panic!("piece {piece:?} missing"))
}

fn byte_id(tokenizer: &Tokenizer, byte: u8) -> u32 {
    id(tokenizer, &format!("<0x{byte:02X}>"))
}

fn raw_table(tokenizer: &Tokenizer) -> Vec<Vec<u8>> {
    (0..tokenizer.get_vocab_size(true) as u32)
        .map(|token| token_raw_bytes(tokenizer, token))
        .collect()
}

fn runtime(source: &str) -> GrammarRuntime {
    let grammar = parse(source).expect("parse grammar");
    let root = grammar.rule_id("root").expect("root rule");
    GrammarRuntime::new(grammar, root).expect("grammar runtime")
}

fn decode_fragments(tokenizer: &Tokenizer, ids: &[u32]) -> (Vec<String>, String) {
    let mut decoder = TokenTextDecoder::default();
    let fragments: Vec<String> = ids.iter().map(|&t| decoder.push(tokenizer, t)).collect();
    (fragments, decoder.finish())
}

#[test]
fn gemma_byte_fallback_tokens_carry_raw_bytes() {
    let tokenizer = gemma_like_tokenizer();
    let lone = byte_id(&tokenizer, 0x82);
    // The lossy single-token rendering the grammar used to see (#553).
    assert_eq!(tokenizer.decode(&[lone], false).unwrap(), REPLACEMENT);
    assert_eq!(token_raw_bytes(&tokenizer, lone), vec![0x82]);
    assert_eq!(
        token_raw_bytes(&tokenizer, byte_id(&tokenizer, 0xE2)),
        vec![0xE2]
    );
    assert_eq!(token_raw_bytes(&tokenizer, byte_id(&tokenizer, b'A')), b"A");
    assert_eq!(
        token_raw_bytes(&tokenizer, id(&tokenizer, "\u{2581}a")),
        b" a"
    );
    // A vocabulary piece that literally spells U+FFFD is real text.
    assert_eq!(
        token_raw_bytes(&tokenizer, id(&tokenizer, REPLACEMENT)),
        REPLACEMENT.as_bytes()
    );
}

#[test]
fn byte_tokens_forming_a_character_decode_once_complete() {
    let tokenizer = gemma_like_tokenizer();
    // "a₂2": U+2082 is E2 82 82, delivered as three byte-fallback tokens.
    let ids = [
        id(&tokenizer, "a"),
        byte_id(&tokenizer, 0xE2),
        byte_id(&tokenizer, 0x82),
        byte_id(&tokenizer, 0x82),
        id(&tokenizer, "2"),
    ];
    let (fragments, tail) = decode_fragments(&tokenizer, &ids);
    // SSE fragments: nothing is emitted until the character is complete.
    assert_eq!(fragments, ["a", "", "", "\u{2082}", "2"]);
    assert_eq!(tail, "");
    // Unary text: the concatenation equals the whole-sequence decode.
    let unary: String = fragments.concat() + &tail;
    assert_eq!(unary, "a\u{2082}2");
    assert_eq!(unary, tokenizer.decode(&ids, false).unwrap());
    assert!(!unary.contains(REPLACEMENT));
}

#[test]
fn invalid_and_truncated_byte_runs_render_replacement() {
    let tokenizer = gemma_like_tokenizer();
    let lone = [
        id(&tokenizer, "3"),
        byte_id(&tokenizer, 0x82),
        id(&tokenizer, "2"),
    ];
    let (fragments, tail) = decode_fragments(&tokenizer, &lone);
    assert_eq!(fragments, ["3", REPLACEMENT, "2"]);
    assert_eq!(tail, "");

    let interrupted = [byte_id(&tokenizer, 0xE2), id(&tokenizer, "a")];
    let (fragments, tail) = decode_fragments(&tokenizer, &interrupted);
    assert_eq!(fragments, ["", "\u{FFFD}a"]);
    assert_eq!(tail, "");

    let truncated = [
        id(&tokenizer, "a"),
        byte_id(&tokenizer, 0xE2),
        byte_id(&tokenizer, 0x82),
    ];
    let (fragments, tail) = decode_fragments(&tokenizer, &truncated);
    assert_eq!(fragments, ["a", "", ""]);
    assert_eq!(tail, REPLACEMENT);

    let mut decoder = TokenTextDecoder::default();
    assert_eq!(decoder.push(&tokenizer, byte_id(&tokenizer, 0xE2)), "");
    decoder.discard();
    assert_eq!(decoder.finish(), "");
}

#[test]
fn grammar_rejects_lone_continuation_byte_inside_json_string() {
    let tokenizer = gemma_like_tokenizer();
    let lone = byte_id(&tokenizer, 0x82);
    let lead = byte_id(&tokenizer, 0xE2);
    let two = id(&tokenizer, "2");
    let table = raw_table(&tokenizer);
    let mut grammar = runtime("root ::= \"\\\"\" [^\"\\\\]* \"\\\"\"\n");
    assert!(grammar.accept_bytes(b"\"(1-3"));

    // The pre-fix lossy table let the byte through as U+FFFD.
    let lossy: Vec<Vec<u8>> = (0..table.len() as u32)
        .map(|t| {
            tokenizer
                .decode(&[t], false)
                .unwrap_or_default()
                .into_bytes()
        })
        .collect();
    let mut logits = vec![1.0_f32; table.len()];
    mask_invalid_tokens_with_eog(&grammar, &lossy, &[], &mut logits);
    assert!(logits[lone as usize].is_finite());

    let mut logits = vec![1.0_f32; table.len()];
    mask_invalid_tokens_with_eog(&grammar, &table, &[], &mut logits);
    assert!(
        !logits[lone as usize].is_finite(),
        "lone 0x82 must be masked"
    );
    assert!(logits[two as usize].is_finite());
    assert!(
        logits[lead as usize].is_finite(),
        "a lead byte may start a character"
    );

    // Greedy: the model's top choice is the lone byte; the grammar picks "2".
    let mut logits = vec![0.0_f32; table.len()];
    logits[lone as usize] = 10.0;
    logits[two as usize] = 5.0;
    assert_eq!(
        sample_greedy_valid_token(&mut logits, &[], 1.0, &grammar, &table, &[]),
        two
    );

    // After a lead byte only a continuation byte may follow.
    assert!(grammar.accept_token(lead, &table[lead as usize]));
    let mut logits = vec![1.0_f32; table.len()];
    mask_invalid_tokens_with_eog(&grammar, &table, &[], &mut logits);
    assert!(logits[lone as usize].is_finite());
    assert!(!logits[two as usize].is_finite());
}

#[test]
fn gemma_tool_grammar_rejects_lone_continuation_byte_in_string_argument() {
    use crate::serve::api::registry::{GrammarShape, GEMMA4};
    let tokenizer = gemma_like_tokenizer();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
        "additionalProperties": false
    });
    let source = GEMMA4
        .tool_call_gbnf("write_file", &schema, GrammarShape::SingleBody)
        .expect("Gemma tool grammar");
    let mut grammar = runtime(&source);
    assert!(grammar.accept_bytes(b"call:write_file{content:<|\"|>(1-3"));
    let table = raw_table(&tokenizer);
    let lone = byte_id(&tokenizer, 0x82) as usize;
    let mut lossy = table.clone();
    lossy[lone] = tokenizer
        .decode(&[lone as u32], false)
        .unwrap()
        .into_bytes();
    let mut logits = vec![1.0_f32; table.len()];
    mask_invalid_tokens_with_eog(&grammar, &lossy, &[], &mut logits);
    assert!(
        logits[lone].is_finite(),
        "the lossy U+FFFD rendering was accepted"
    );

    let mut logits = vec![1.0_f32; table.len()];
    mask_invalid_tokens_with_eog(&grammar, &table, &[], &mut logits);
    assert!(!logits[lone].is_finite());
    assert!(logits[id(&tokenizer, "2") as usize].is_finite());
}

#[test]
fn byte_level_pieces_carry_raw_bytes() {
    // Inverse GPT-2 map: a bijection from the ByteLevel alphabet onto bytes.
    let alphabet = ByteLevel::alphabet();
    assert_eq!(alphabet.len(), 256);
    let bytes: HashSet<u8> = alphabet
        .iter()
        .map(|&c| byte_level_byte(c).expect("alphabet char maps"))
        .collect();
    assert_eq!(bytes.len(), 256);
    assert_eq!(byte_level_byte('\u{0120}'), Some(b' '));
    assert_eq!(byte_level_byte('\u{010A}'), Some(b'\n'));
    assert_eq!(byte_level_byte('\u{0124}'), Some(0x82));

    // "âĤ" is the byte-level spelling of E2 82 (an incomplete U+2082).
    let pieces = ["a", "\u{00E2}\u{0124}", "\u{0124}"];
    let vocab: AHashMap<String, u32> = pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| (piece.to_string(), id as u32))
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("byte-level BPE");
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    assert_eq!(tokenizer.decode(&[1], false).unwrap(), REPLACEMENT);
    assert_eq!(token_raw_bytes(&tokenizer, 0), b"a");
    assert_eq!(token_raw_bytes(&tokenizer, 1), vec![0xE2, 0x82]);
    assert_eq!(token_raw_bytes(&tokenizer, 2), vec![0x82]);
    let (fragments, tail) = decode_fragments(&tokenizer, &[0, 1, 2, 0]);
    assert_eq!(fragments, ["a", "", "\u{2082}", "a"]);
    assert_eq!(tail, "");
}

/// Real artifact (ignored): `HF2Q_GEMMA4_GGUF` names a Gemma 4 GGUF. Token
/// 368 is the `<0x82>` piece the i13 replay sampled inside `(1-3?2)`.
#[test]
#[ignore = "requires HF2Q_GEMMA4_GGUF"]
fn real_gemma4_vocab_byte_tokens_carry_raw_bytes() {
    let path = std::env::var("HF2Q_GEMMA4_GGUF").expect("HF2Q_GEMMA4_GGUF");
    let gguf = mlx_native::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open GGUF");
    let tokenizer =
        crate::inference::models::gemma4::tokenizer::build_tokenizer_from_gguf(&gguf).unwrap();
    assert_eq!(tokenizer.id_to_token(368).as_deref(), Some("<0x82>"));
    assert_eq!(tokenizer.decode(&[368], false).unwrap(), REPLACEMENT);
    assert_eq!(token_raw_bytes(&tokenizer, 368), vec![0x82]);
    let literal = tokenizer
        .token_to_id(REPLACEMENT)
        .expect("literal U+FFFD piece");
    assert_eq!(token_raw_bytes(&tokenizer, literal), REPLACEMENT.as_bytes());
    let table = raw_table(&tokenizer);
    let lossy = (0..table.len() as u32)
        .filter(|&t| table[t as usize] != tokenizer.decode(&[t], false).unwrap().into_bytes())
        .count();
    // Exactly the 128 non-ASCII byte-fallback pieces differ from decode.
    assert_eq!(lossy, 128);
}
