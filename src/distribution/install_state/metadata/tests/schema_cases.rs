use super::*;

#[test]
fn generation_and_selector_have_exact_canonical_v2_bytes() {
    let candidate = candidate(
        "2026-08-17T20:00:00.123456789Z",
        "2026-08-17T20:00:01.123456789Z",
        2,
        3,
    );
    let receipt = MetadataGenerationReceiptV2::new(1, None, &candidate).expect("receipt");
    let bytes = receipt.to_bytes().expect("serialize");
    assert_eq!(
        bytes,
        include_bytes!("../testdata/generation-v2.json"),
        "schema drift requires an explicit v3 wire contract"
    );
    assert_eq!(
        MetadataGenerationReceiptV2::parse(&bytes)
            .expect("parse")
            .to_bytes()
            .expect("reserialize"),
        bytes
    );
    assert!(receipt.matches_candidate(&candidate));
    assert_eq!(
        receipt.expected_root_names(),
        vec!["00000000000000000002.root.json"]
    );

    let selector = MetadataSelectorV2::new(1, receipt.digest().expect("digest")).expect("selector");
    let selector_bytes = selector.to_bytes().expect("selector bytes");
    assert_eq!(
        selector_bytes,
        include_bytes!("../testdata/selector-v2.json"),
        "selector drift requires an explicit v3 wire contract"
    );
    assert_eq!(
        MetadataSelectorV2::parse(&selector_bytes)
            .expect("selector parse")
            .to_bytes()
            .expect("selector reserialize"),
        selector_bytes
    );
}

#[test]
fn populated_floor_reset_has_exact_canonical_v2_bytes() {
    let prior_candidate = candidate(
        "2026-08-17T20:00:00.123456789Z",
        "2026-08-17T20:00:01.123456789Z",
        2,
        100,
    );
    let prior = MetadataGenerationReceiptV2::new(1, None, &prior_candidate).expect("prior");
    let predecessor = prior.digest().expect("predecessor digest");
    let mut recovered = candidate(
        "2026-08-17T20:00:01.123456789Z",
        "2026-08-17T20:00:02.123456789Z",
        3,
        2,
    );
    recovered.set_timestamp_snapshot_floor_reset_for_test(Some(2));
    let receipt = MetadataGenerationReceiptV2::new(2, Some(predecessor), &recovered)
        .expect("populated reset receipt");
    let receipt_bytes = receipt.to_bytes().expect("receipt bytes");
    assert_eq!(
        receipt_bytes,
        include_bytes!("../testdata/generation-v2-floor-reset.json"),
        "populated reset schema drift requires an explicit v3 wire contract"
    );
    let selector = MetadataSelectorV2::new(2, receipt.digest().expect("receipt digest"))
        .expect("reset selector");
    assert_eq!(
        selector.to_bytes().expect("selector bytes"),
        include_bytes!("../testdata/selector-v2-floor-reset.json"),
        "populated reset selector drift requires an explicit v3 wire contract"
    );
}

#[test]
fn prepublication_v1_receipt_and_selector_are_explicitly_rejected() {
    assert!(matches!(
        MetadataGenerationReceiptV2::parse(include_bytes!(
            "../testdata/generation-v1-prepublication.json"
        )),
        Err(MetadataJournalError::Invalid(
            "generation receipt envelope is unsupported"
        ))
    ));
    assert!(matches!(
        MetadataSelectorV2::parse(include_bytes!(
            "../testdata/selector-v1-prepublication.json"
        )),
        Err(MetadataJournalError::Invalid(
            "metadata selector envelope is unsupported"
        ))
    ));
}

#[test]
fn successor_enforces_clock_and_role_floors() {
    let prior_candidate = candidate("2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 2, 3);
    let prior = MetadataGenerationReceiptV2::new(1, None, &prior_candidate).expect("prior");
    let digest = prior.digest().expect("digest");

    let next_candidate = candidate("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4);
    let next = MetadataGenerationReceiptV2::new(2, Some(digest.clone()), &next_candidate)
        .expect("successor");
    next.validate_successor(&prior, &digest)
        .expect("valid successor");

    let backward_candidate = candidate("2026-08-17T19:59:59Z", "2026-08-17T20:00:00Z", 3, 4);
    let backward = MetadataGenerationReceiptV2::new(2, Some(digest.clone()), &backward_candidate)
        .expect("structurally valid receipt");
    assert!(matches!(
        backward.validate_successor(&prior, &digest),
        Err(MetadataJournalError::Invalid(
            "metadata verification clock floor moved backward"
        ))
    ));

    let rollback_candidate = candidate("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 2, 2);
    let rollback = MetadataGenerationReceiptV2::new(2, Some(digest.clone()), &rollback_candidate)
        .expect("structurally valid receipt");
    assert!(matches!(
        rollback.validate_successor(&prior, &digest),
        Err(MetadataJournalError::Invalid(
            "metadata role floor moved backward or equivocated"
        ))
    ));

    let mut changed_history_candidate =
        candidate("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4);
    changed_history_candidate.replace_root_for_test(0, role("2.root.json", 2, "different-root-2"));
    let changed_history =
        MetadataGenerationReceiptV2::new(2, Some(digest.clone()), &changed_history_candidate)
            .expect("structurally valid receipt");
    assert!(matches!(
        changed_history.validate_successor(&prior, &digest),
        Err(MetadataJournalError::Invalid(
            "metadata root history changed below its trusted floor"
        ))
    ));

    let mut reset_candidate = candidate("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4);
    reset_candidate.set_timestamp_snapshot_floor_reset_for_test(Some(2));
    let reset = MetadataGenerationReceiptV2::new(2, Some(digest.clone()), &reset_candidate)
        .expect("sealed reset receipt");
    reset
        .validate_successor(&prior, &digest)
        .expect("reset is bound to the appended root transition");
    assert!(reset.matches_candidate(&reset_candidate));
    reset_candidate.set_timestamp_snapshot_floor_reset_for_test(None);
    assert!(
        !reset.matches_candidate(&reset_candidate),
        "receipt reset evidence cannot be added or removed without changing the sealed candidate"
    );

    assert!(matches!(
        reset.validate_timestamp_snapshot_floor_reset(
            reset_candidate.anchor_root().bytes(),
            reset_candidate.root_chain(),
            reset_candidate.trusted_root().bytes(),
            false,
        ),
        Err(MetadataJournalError::Invalid(
            "online-role floor reset lacks an authenticated key rotation"
        ))
    ));

    let mut wrong_source = candidate("2026-08-17T20:00:01Z", "2026-08-17T20:00:02Z", 3, 4);
    wrong_source.set_timestamp_snapshot_floor_reset_for_test(Some(1));
    let wrong_source = MetadataGenerationReceiptV2::new(2, Some(digest.clone()), &wrong_source)
        .expect("structurally valid wrong-source receipt");
    assert!(matches!(
        wrong_source.validate_successor(&prior, &digest),
        Err(MetadataJournalError::Invalid(
            "online-role floor reset is not bound to the new root transition"
        ))
    ));
}

#[test]
fn hostile_receipt_encodings_fail_closed() {
    let candidate = candidate("2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 1, 1);
    let bytes = MetadataGenerationReceiptV2::new(1, None, &candidate)
        .expect("receipt")
        .to_bytes()
        .expect("bytes");

    let mut noncanonical = bytes.clone();
    noncanonical.insert(0, b' ');
    assert!(MetadataGenerationReceiptV2::parse(&noncanonical).is_err());

    let trailing = [bytes.as_slice(), b"{}"].concat();
    assert!(MetadataGenerationReceiptV2::parse(&trailing).is_err());

    let canonical = String::from_utf8(bytes).expect("UTF-8");
    let duplicate = canonical.replacen(
        "\"schema_version\":2,",
        "\"schema_version\":2,\"schema_version\":2,",
        1,
    );
    assert!(MetadataGenerationReceiptV2::parse(duplicate.as_bytes()).is_err());

    let nested_unknown = canonical.replace(
        "\"anchor_root\":{\"request_name\"",
        "\"anchor_root\":{\"unexpected\":true,\"request_name\"",
    );
    assert!(MetadataGenerationReceiptV2::parse(nested_unknown.as_bytes()).is_err());

    let oversized = vec![b' '; MAX_GENERATION_RECEIPT_BYTES + 1];
    assert!(matches!(
        MetadataGenerationReceiptV2::parse(&oversized),
        Err(MetadataJournalError::Invalid(
            "generation receipt exceeds its input bound"
        ))
    ));

    let selector = MetadataSelectorV2::new(1, "a".repeat(64))
        .expect("selector")
        .to_bytes()
        .expect("selector bytes");
    let selector_text = String::from_utf8(selector.clone()).expect("UTF-8 selector");
    for hostile in [
        selector_text.replacen(
            "\"schema_version\":2,",
            "\"schema_version\":2,\"schema_version\":2,",
            1,
        ),
        selector_text.replacen(
            "\"schema_version\":2,",
            "\"schema_version\":2,\"unexpected\":true,",
            1,
        ),
        format!("{selector_text}{{}}"),
    ] {
        assert!(MetadataSelectorV2::parse(hostile.as_bytes()).is_err());
    }
    assert!(matches!(
        MetadataSelectorV2::parse(&vec![b' '; MAX_SELECTOR_BYTES + 1]),
        Err(MetadataJournalError::Invalid(
            "metadata selector exceeds its input bound"
        ))
    ));
}

#[test]
fn root_history_lifetime_bound_is_enforced_without_limiting_updates() {
    let maximum = candidate(
        "2026-08-17T20:00:00Z",
        "2026-08-17T20:00:01Z",
        MAX_ROOT_CHAIN as u64 + 1,
        1,
    );
    assert_eq!(
        MetadataGenerationReceiptV2::new(1, None, &maximum)
            .expect("maximum legal root history")
            .expected_root_names()
            .len(),
        MAX_ROOT_CHAIN
    );

    let candidate = candidate(
        "2026-08-17T20:00:00Z",
        "2026-08-17T20:00:01Z",
        MAX_ROOT_CHAIN as u64 + 2,
        1,
    );
    assert!(matches!(
        MetadataGenerationReceiptV2::new(1, None, &candidate),
        Err(MetadataJournalError::Invalid(
            "root history exceeds the v2 lifetime bound"
        ))
    ));
}

#[test]
fn journal_commits_more_than_the_discarded_spikes_update_cap() {
    let parent = tempfile::tempdir().expect("tempdir");
    let root = test_root(&parent);
    for sequence in 1..=1_025 {
        assert_eq!(
            commit_candidate_for_test(
                authorization(&root),
                candidate_at(
                    &root,
                    "2026-08-17T20:00:01Z",
                    "2026-08-17T20:00:01Z",
                    1,
                    sequence,
                ),
                FaultPlan::default(),
            )
            .unwrap_or_else(|error| panic!("generation {sequence} failed: {error}")),
            MetadataCommitOutcome::Committed { sequence }
        );
    }
    assert_eq!(
        read_selected(&authorization(&root))
            .expect("read final selection")
            .expect("selected generation")
            .sequence,
        1_025
    );
    assert_eq!(
        std::fs::read_dir(root.join("update/metadata/generations"))
            .expect("bounded generation inventory")
            .count(),
        1
    );
}

#[test]
fn request_name_must_bind_the_parsed_role_version() {
    let mut candidate = candidate("2026-08-17T20:00:00Z", "2026-08-17T20:00:01Z", 1, 2);
    candidate
        .snapshot_mut_for_test()
        .set_request_name_for_test("999.snapshot.json");
    assert!(matches!(
        MetadataGenerationReceiptV2::new(1, None, &candidate),
        Err(MetadataJournalError::Invalid(
            "metadata request name and role version disagree"
        ))
    ));
}
