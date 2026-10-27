//! Test-only fixture composition; production source must retain its exact hash.
use super::prefix_post_oracle_tests::hash;

const DECLARATIONS: &[&str] = &[
    "\n#[cfg(test)]\n#[path = \"handlers_capacity_fixture_source_tests.rs\"]\nmod capacity_fixture_source_tests;\n",
    "\n#[cfg(test)]\n#[path = \"handlers_qwen36_largest_prompt_tests.rs\"]\nmod qwen36_largest_prompt_tests;\n",
    "\n#[cfg(test)]\n#[path = \"handlers_qwen38_largest_prompt_tests.rs\"]\nmod qwen38_largest_prompt_tests;\n",
    "\n#[cfg(test)]\n#[path = \"handlers_tool_result_capacity_tokens_tests.rs\"]\nmod tool_result_capacity_tokens_tests;\n",
    "\n#[cfg(test)]\n#[path = \"handlers_gemma_largest_prompt_tests.rs\"]\nmod gemma_largest_prompt_tests;\n",
    "\n#[cfg(test)]\n#[path = \"handlers_gemma_full_input_post_oracle_tests.rs\"]\nmod gemma_full_input_post_oracle_tests;\n",
    "\n#[cfg(test)]\n#[path = \"handlers_gemma_matched_tokens_tests.rs\"]\nmod gemma_matched_tokens_tests;\n",
];

/// Remove only these exact appended test declarations, stopping at the frozen
/// source hash. Unknown suffixes and any production-body mutation fail closed.
/// Existing historical seven-wire quality oracles do not use this utility.
pub(super) fn source_at_base<'a>(
    mut source: &'a str,
    expected_hash: &str,
) -> Result<&'a str, &'static str> {
    loop {
        if hash(source.as_bytes()) == expected_hash {
            return Ok(source);
        }
        source = DECLARATIONS
            .iter()
            .find_map(|declaration| source.strip_suffix(declaration))
            .ok_or("source differs beyond the exact known capacity-fixture declarations")?;
    }
}

#[test]
fn capacity_fixture_source_composes_only_exact_declarations() {
    let base = "fn production() { keep_original_behavior(); }\n";
    let expected = hash(base.as_bytes());
    let appended = DECLARATIONS
        .iter()
        .fold(base.to_owned(), |mut s, declaration| {
            s.push_str(declaration);
            s
        });
    assert_eq!(source_at_base(&appended, &expected), Ok(base));
    let intermediate = format!("{base}{}", DECLARATIONS[5]);
    let later = format!("{intermediate}{}", DECLARATIONS[0]);
    assert_eq!(
        source_at_base(&later, &hash(intermediate.as_bytes())),
        Ok(intermediate.as_str())
    );
}

#[test]
fn capacity_fixture_source_rejects_unknown_declarations_and_body_changes() {
    let base = "fn production() { keep_original_behavior(); }\n";
    let expected = hash(base.as_bytes());
    let unknown = format!(
        "{base}\n#[cfg(test)]\nmod unknown_fixture;\n{}",
        DECLARATIONS[0]
    );
    assert!(source_at_base(&unknown, &expected).is_err());
    let changed = format!(
        "fn production() {{ changed_behavior(); }}\n{}",
        DECLARATIONS[0]
    );
    assert!(source_at_base(&changed, &expected).is_err());
}
