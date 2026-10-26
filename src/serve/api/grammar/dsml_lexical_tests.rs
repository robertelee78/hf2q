//! Actual emitted grammar, acceptance and collective-mask controls; no device.
use super::{dsml_lexical::*, mask::mask_invalid_tokens, parser::parse, sampler::GrammarRuntime};
use crate::serve::api::registry;
const CLOSE_TEXT: &str = "</｜DSML｜parameter>";
fn source(min: u64, max: Option<u64>, json: bool, structural: bool) -> String {
    let mut rules = Vec::new();
    let body = registry::dsml_string_for_test(&mut rules, min, max, json).unwrap();
    let root = if json {
        format!("\"\\\"\" {body} \"\\\"\"")
    } else {
        body
    };
    let suffix = if structural {
        format!(
            " {} \"END\"",
            super::super::registry::gbnf_literal_for_test(CLOSE_TEXT)
        )
    } else {
        String::new()
    };
    format!(
        "root ::= {root}{suffix}\n{}",
        rules
            .into_iter()
            .map(|(n, b)| format!("{n} ::= {b}\n"))
            .collect::<String>()
    )
}
fn runtime(src: &str) -> GrammarRuntime {
    let g = parse(src).unwrap();
    let id = g.rule_id("root").unwrap();
    GrammarRuntime::new(g, id).unwrap()
}
fn accepts(src: &str, text: &str) -> bool {
    let mut rt = runtime(src);
    rt.accept_bytes(text.as_bytes()) && rt.is_accepted()
}
#[test]
fn dsml_compact_exact_bounds_unicode_escape_and_delimiter() {
    let src = source(2, Some(4), false, true);
    for data in ["ab", "😀é", "<>", "a\nb", "\\<", "a\\\\"] {
        assert!(accepts(&src, &format!("{data}{CLOSE_TEXT}END")), "{data:?}");
    }
    for data in ["a", "abcde", "😀é中xy", "a\\", "\\\n"] {
        assert!(
            !accepts(&src, &format!("{data}{CLOSE_TEXT}END")),
            "{data:?}"
        );
    }
    let src = source(0, Some(256), false, true);
    assert!(accepts(
        &src,
        &format!("fmt::Formatter<'_> Vec::<T> a < b\n{CLOSE_TEXT}END")
    ));
    assert!(!accepts(&src, &format!("x{CLOSE_TEXT}y{CLOSE_TEXT}END")));
    assert!(!accepts(&src, &format!("\\{CLOSE_TEXT}x{CLOSE_TEXT}END")));
    let json = source(1, Some(2), true, false);
    for data in ["😀", r"\u003c", r"\n", r#"\""#, r"\\", "<>"] {
        assert!(accepts(&json, &format!("\"{data}\"")), "{data:?}");
    }
    for data in ["", "abc", "\n", "\"", "\\", r"\q"] {
        assert!(!accepts(&json, &format!("\"{data}\"")), "{data:?}");
    }
    let json = source(0, Some(256), true, false);
    assert!(!accepts(&json, &format!("\"{CLOSE_TEXT}\"")));
    assert!(accepts(
        &json,
        &format!("\"{}\"", CLOSE_TEXT.replace('<', r"\u003c"))
    ));
}
#[test]
fn dsml_compact_4096_and_65536_bounds_no_cross_product() {
    for n in [4096, 65536] {
        for json in [false, true] {
            let src = source(n, Some(n), json, false);
            let g = parse(&src).unwrap();
            assert!(
                g.rules.len() < 10_000,
                "compact existing repetition, not length x delimiter"
            );
            let quote = if json { "\"" } else { "" };
            assert!(accepts(
                &src,
                &format!("{quote}{}{quote}", "a".repeat(n as usize))
            ));
            assert!(!accepts(
                &src,
                &format!("{quote}{}{quote}", "a".repeat(n as usize - 1))
            ));
            assert!(!accepts(
                &src,
                &format!("{quote}{}{quote}", "a".repeat(n as usize + 1))
            ));
        }
    }
}
#[test]
fn dsml_compact_collective_mask_equals_acceptance_across_splits() {
    let src = source(0, Some(256), true, false);
    let prefix = format!("\"{}", CLOSE_TEXT.trim_end_matches('>'));
    let mut rt = runtime(&src);
    assert!(rt.accept_bytes(prefix.as_bytes()));
    let tokens = vec![
        b">".to_vec(),
        b"x".to_vec(),
        b"x\"".to_vec(),
        "😀".as_bytes().to_vec(),
        b">x\"".to_vec(),
    ];
    let mut logits = vec![1.0; tokens.len()];
    mask_invalid_tokens(&rt, &tokens, &mut logits);
    for (i, bytes) in tokens.iter().enumerate() {
        let mut probe = rt.clone();
        let alive = probe.accept_token(i as u32, bytes);
        assert_eq!(logits[i].is_finite(), alive, "candidate{i}");
    }
    assert!(!logits[0].is_finite());
    assert!(logits[1].is_finite());
    let mut greedy = vec![10.0, 1.0];
    assert_eq!(
        super::mask::sample_greedy_valid_token(&mut greedy, &[], 1.0, &rt, &tokens[..2], &[]),
        1
    );
    // Every byte split includes breaks inside the two fullwidth-bar scalars.
    let bad = format!("\"{CLOSE_TEXT}\"");
    for split in 0..=bad.len() {
        let mut rt = runtime(&src);
        let first = rt.accept_token(0, &bad.as_bytes()[..split]);
        let tail = bad.as_bytes()[split..].to_vec();
        let mut logits = vec![1.0];
        mask_invalid_tokens(&rt, &[tail.clone()], &mut logits);
        let mut bytes_probe = rt.clone();
        let bytes_ok = bytes_probe.accept_bytes(&tail);
        let second = rt.accept_token(1, &tail);
        assert_eq!(second, bytes_ok, "token/bytes split{split}");
        if !tail.is_empty() {
            assert_eq!(logits[0].is_finite(), second, "mask split{split}");
        }
        assert!(!(first && second && rt.is_accepted()), "split{split}");
    }
    let raw = source(0, Some(256), false, true);
    let mut structural = runtime(&raw);
    assert!(structural.accept_bytes(b"x"));
    let close = format!("{CLOSE_TEXT}END").into_bytes();
    let mut logits = vec![1.0];
    mask_invalid_tokens(&structural, &[close.clone()], &mut logits);
    assert!(logits[0].is_finite());
    assert!(structural.accept_token(0, &close));
    assert!(structural.is_accepted());
    let mut escaped = runtime(&raw);
    assert!(escaped.accept_bytes(b"x\\"));
    let mut logits = vec![1.0];
    mask_invalid_tokens(&escaped, &[close], &mut logits);
    assert!(!logits[0].is_finite());
    rt.trigger(); // Duplicate active router notification cannot erase prefix.
    assert!(!rt.accept_bytes(b">"));
    let mut fresh = runtime(&src);
    assert!(fresh.accept_bytes(b"\">\""));
    assert!(fresh.is_accepted());
    // Epsilon RAW_END sees EACH candidate's AFTER-consumed scan. A single
    // shared expansion would make '\\' and ordinary characters disagree.
    for (bound, pieces) in [
        (
            1,
            vec![b"\\".to_vec(), b"a".to_vec(), "😀".as_bytes().to_vec()],
        ),
        (
            2,
            vec![
                b"\\x".to_vec(),
                b"x\\".to_vec(),
                b"\\\\".to_vec(),
                b"\\\n".to_vec(),
            ],
        ),
    ] {
        let exact = source(bound, Some(bound), false, false);
        let rt = runtime(&exact);
        let mut logits = vec![1.0; pieces.len()];
        mask_invalid_tokens(&rt, &pieces, &mut logits);
        for (i, bytes) in pieces.iter().enumerate() {
            let mut tok = rt.clone();
            let mut raw = rt.clone();
            assert_eq!(
                logits[i].is_finite(),
                tok.accept_token(i as u32, bytes),
                "bound{bound}/token{i}"
            );
            assert_eq!(
                logits[i].is_finite(),
                raw.accept_bytes(bytes),
                "bound{bound}/bytes{i}"
            );
        }
        assert!(!logits[usize::from(bound == 2)].is_finite());
        assert!(logits[usize::from(bound == 1)].is_finite());
    }
}
#[test]
fn dsml_compact_lazy_and_serialized_namespace_keep_guard() {
    let src = source(0, Some(256), true, false);
    let prefix = format!("OPEN\"{}", CLOSE_TEXT.trim_end_matches('>'));
    let mut rt = runtime(&src);
    rt.set_lazy_trigger(b"OPEN");
    assert!(rt.accept_token(0, prefix.as_bytes()));
    assert!(!rt.is_awaiting_trigger());
    let mut logits = vec![1.0, 1.0];
    mask_invalid_tokens(&rt, &[b">".to_vec(), b"x".to_vec()], &mut logits);
    assert!(!logits[0].is_finite());
    assert!(logits[1].is_finite());
    let mut clone = rt.clone();
    assert!(!clone.accept_token(1, b">"));
    assert!(rt.accept_token(1, b"x\""));
    let parsed = parse(&src).unwrap();
    let renamed = super::serialize::rename_rules(&parsed, |n| format!("fn-12-variant-3-{n}"));
    let wire = super::serialize::serialize(&renamed);
    let g = parse(&wire).unwrap();
    let id = g.rule_id("fn-12-variant-3-root").unwrap();
    let mut rt = GrammarRuntime::new(g, id).unwrap();
    assert!(!rt.accept_bytes(format!("\"{CLOSE_TEXT}\"").as_bytes()));
}
#[test]
fn dsml_compact_reserved_rule_collision_refuses_no_guard_default() {
    let bad = format!("root ::= {RAW_NAME}\n{RAW_NAME} ::= [a]\n");
    let g = parse(&bad).unwrap();
    let id = g.rule_id("root").unwrap();
    assert!(GrammarRuntime::new(g, id).is_none());
    let src = source(0, Some(4), false, false);
    let renamed = super::serialize::rename_rules(&parse(&src).unwrap(), |n| format!("unknown-{n}"));
    let id = renamed.rule_id("unknown-root").unwrap();
    assert!(GrammarRuntime::new(renamed, id).is_none());
    let mut dangling = runtime(&source(0, Some(4), false, false));
    assert!(dangling.accept_bytes(b"\\"));
    assert!(!dangling.is_accepted());
    assert!(!dangling.accept_eog());
    let literal = registry::gbnf_literal_for_test("x\\");
    let mixed = source(0, Some(4), false, false).replacen(
        "root ::= ",
        &format!("root ::= {literal} | "),
        1,
    );
    let mut rt = runtime(&mixed);
    assert!(rt.accept_bytes(b"x\\"));
    assert!(rt.is_accepted());
    assert!(rt.accept_eog());
    let bad = format!("root ::= {END_NAME}\n{END_NAME} ::= [a]\n");
    let g = parse(&bad).unwrap();
    let id = g.rule_id("root").unwrap();
    assert!(GrammarRuntime::new(g, id).is_none());
    for token in ["<[7]>", "!<[7]>", "<[*]>", "!<[1,2]>"] {
        let guarded = format!("root ::= {token} {END_NAME}\n{END_NAME} ::= {END_BODY}\n");
        let g = parse(&guarded).unwrap();
        let id = g.rule_id("root").unwrap();
        assert!(
            GrammarRuntime::new(g, id).is_none(),
            "private DSML token capability refused: {token}"
        );
        let plain = format!("root ::= {token}\n");
        let g = parse(&plain).unwrap();
        let id = g.rule_id("root").unwrap();
        assert!(
            GrammarRuntime::new(g, id).is_some(),
            "ordinary token grammar unchanged: {token}"
        );
    }
    let mut plain = runtime("root ::= [^x]*\n");
    assert!(plain.dsml_guard.is_none());
    assert!(plain.accept_bytes(CLOSE_TEXT.as_bytes()));
    assert!(plain.is_accepted());
}
