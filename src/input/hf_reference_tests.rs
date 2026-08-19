use super::hf_reference::{HfModelReference, HfRepositoryType};

const SHA: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

#[test]
fn equivalent_model_references_share_one_canonical_identity() {
    let cases = [
        ("Qwen/Qwen3.8-27B", None, None, None),
        (
            "https://huggingface.co/Qwen/Qwen3.8-27B",
            None,
            None,
            None,
        ),
        (
            "https://huggingface.co/Qwen/Qwen3.8-27B/",
            None,
            None,
            None,
        ),
        (
            "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
            None,
            Some("main"),
            None,
        ),
        (
            "https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json",
            None,
            Some("main"),
            Some("config.json"),
        ),
        (
            "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/processor/preprocessor_config.json",
            None,
            Some("main"),
            Some("processor/preprocessor_config.json"),
        ),
    ];

    for (input, explicit_revision, requested_revision, filename) in cases {
        let reference = HfModelReference::parse(input, explicit_revision).unwrap();
        assert_eq!(reference.original(), input);
        assert_eq!(reference.repo_id(), "Qwen/Qwen3.8-27B");
        assert_eq!(reference.repository_type(), HfRepositoryType::Model);
        assert_eq!(
            reference.canonical_url(),
            "https://huggingface.co/Qwen/Qwen3.8-27B"
        );
        assert_eq!(reference.requested_revision(), requested_revision);
        assert_eq!(reference.filename(), filename);
    }
}

#[test]
fn explicit_and_embedded_revisions_must_agree() {
    let reference = HfModelReference::parse(
        "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
        Some("main"),
    )
    .unwrap();
    assert_eq!(reference.requested_revision(), Some("main"));

    let error = HfModelReference::parse(
        "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
        Some("refs/pr/7"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not match"));

    let uppercase_sha = SHA.to_ascii_uppercase();
    let reference = HfModelReference::parse(
        &format!("https://huggingface.co/Qwen/Qwen3.8-27B/tree/{uppercase_sha}"),
        Some(SHA),
    )
    .unwrap();
    assert_eq!(reference.requested_revision(), Some(SHA));
}

#[test]
fn resolved_reference_requires_an_exact_immutable_commit() {
    let unresolved = HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    assert!(unresolved.clone().resolve("main").is_err());
    assert!(unresolved.clone().resolve(&"a".repeat(39)).is_err());
    assert!(unresolved.clone().resolve(&"g".repeat(40)).is_err());

    let resolved = unresolved.resolve(&SHA.to_ascii_uppercase()).unwrap();
    assert_eq!(resolved.original(), "Qwen/Qwen3.8-27B");
    assert_eq!(resolved.repo_id(), "Qwen/Qwen3.8-27B");
    assert_eq!(resolved.repository_type(), HfRepositoryType::Model);
    assert_eq!(
        resolved.canonical_url(),
        "https://huggingface.co/Qwen/Qwen3.8-27B"
    );
    assert_eq!(resolved.revision(), SHA);
    assert_eq!(resolved.filename(), None);
}

#[test]
fn valid_single_component_repo_and_encoded_revision_are_supported() {
    let plain = HfModelReference::parse("gpt2", Some("main")).unwrap();
    assert_eq!(plain.repo_id(), "gpt2");
    assert_eq!(plain.requested_revision(), Some("main"));

    let tree = HfModelReference::parse(
        "https://huggingface.co/gpt2/tree/refs%2Fpr%2F7",
        Some("refs/pr/7"),
    )
    .unwrap();
    assert_eq!(tree.repo_id(), "gpt2");
    assert_eq!(tree.requested_revision(), Some("refs/pr/7"));
}

#[test]
fn hostile_urls_and_ambiguous_routes_fail_closed() {
    let invalid = [
        "http://huggingface.co/Qwen/Qwen3.8-27B",
        "https://www.huggingface.co/Qwen/Qwen3.8-27B",
        "https://evil.example/Qwen/Qwen3.8-27B",
        "https://user@huggingface.co/Qwen/Qwen3.8-27B",
        "https://huggingface.co:444/Qwen/Qwen3.8-27B",
        "https://huggingface.co/Qwen/Qwen3.8-27B?download=1",
        "https://huggingface.co/Qwen/Qwen3.8-27B#readme",
        "https://huggingface.co/Qwen/Qwen3.8-27B/tree",
        "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main/extra",
        "https://huggingface.co/Qwen/Qwen3.8-27B/blob/main",
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main",
        "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json",
        "https://huggingface.co/Qwen%2FEvil/Qwen3.8-27B",
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/%2E%2E/secret",
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/a%2Fb",
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/bad%ZZname",
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/truncated%2",
    ];
    for input in invalid {
        assert!(
            HfModelReference::parse(input, None).is_err(),
            "accepted hostile or ambiguous input: {input}"
        );
    }
}

#[test]
fn route_words_are_still_valid_repository_names_without_a_route_tail() {
    for input in [
        "owner/tree",
        "owner/blob",
        "owner/resolve",
        "https://huggingface.co/owner/tree",
        "https://huggingface.co/owner/blob",
        "https://huggingface.co/owner/resolve",
    ] {
        let reference = HfModelReference::parse(input, None).unwrap();
        let expected = input
            .strip_prefix("https://huggingface.co/")
            .unwrap_or(input);
        assert_eq!(reference.repo_id(), expected);
    }
}

#[test]
fn repository_and_revision_grammar_is_bounded_and_canonical() {
    let invalid_repos = [
        "",
        "/repo",
        "owner/",
        "owner/repo/extra",
        "-owner/repo",
        "owner-/repo",
        ".owner/repo",
        "owner./repo",
        "owner/repo.git",
        "owner/re--po",
        "owner/re..po",
        "owner/re po",
        "owner/repo\\name",
        "owner/💥",
    ];
    for input in invalid_repos {
        assert!(
            HfModelReference::parse(input, None).is_err(),
            "accepted invalid repository: {input}"
        );
    }
    assert!(HfModelReference::parse(&format!("owner/{}", "a".repeat(91)), None).is_err());

    let invalid_revisions = [
        "",
        "/main",
        "main/",
        "refs//main",
        "refs/../main",
        "refs\\main",
        "main branch",
        "main?x",
        "main#x",
        "%2F",
    ];
    for revision in invalid_revisions {
        assert!(
            HfModelReference::parse("owner/repo", Some(revision)).is_err(),
            "accepted invalid revision: {revision:?}"
        );
    }
}

#[test]
fn input_size_and_filename_depth_are_bounded_before_allocation_growth() {
    let oversized = format!("owner/{}", "a".repeat(2048));
    assert!(HfModelReference::parse(&oversized, None).is_err());

    let deep_file = (0..65)
        .map(|index| format!("d{index}"))
        .collect::<Vec<_>>()
        .join("/");
    let url = format!("https://huggingface.co/owner/repo/resolve/main/{deep_file}");
    assert!(HfModelReference::parse(&url, None).is_err());
}
