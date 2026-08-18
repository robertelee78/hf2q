use super::*;

#[test]
fn exact_static_corpus_is_pinned_and_authenticates_incrementally() {
    for (bytes, expected) in [
        (
            ROOT,
            "5401ed31f3943848b78c5e2c2998026b65bb7430cd41c9f587ee2c1700ca57c4",
        ),
        (
            TIMESTAMP,
            "774d9fef78ecd45e84ca5316ab87f8fa39cbf07fb6f60d80d940e6aa0a091dfd",
        ),
        (
            SNAPSHOT,
            "585d0e5f57e76acedbe01069d11a799c24f503c4b6064dc2ba077f00d94ed193",
        ),
        (
            TARGETS,
            "e1a2a22d67c7b9af5b291a0950bb26e5262ed85317ac59259cfc27448f2dc88e",
        ),
    ] {
        assert_eq!(hex::encode(Sha256::digest(bytes)), expected);
    }

    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let started = instant("2026-08-18T08:30:00.123456789Z");
    let completed = instant("2026-08-18T08:30:00.223456789Z");
    let candidate = complete_static_transcript(
        begin_from_anchor_for_test(&authorization, &anchor, [started, completed])
            .expect("compiled anchor starts the verifier"),
    );
    assert_eq!(candidate.installation_id(), INSTALLATION_ID);
    assert_eq!(candidate.verification_started_at(), started);
    assert_eq!(candidate.verification_completed_at(), completed);
    assert_eq!(candidate.anchor_root().bytes(), ROOT);
    assert!(candidate.root_chain().is_empty());
    assert_eq!(candidate.timestamp().bytes(), TIMESTAMP);
    assert_eq!(candidate.snapshot().bytes(), SNAPSHOT);
    assert_eq!(candidate.targets().bytes(), TARGETS);
}

#[test]
fn independent_python_tuf_corpus_cross_authenticates() {
    for (bytes, expected) in [
        (
            PYTHON_ROOT_V1,
            "495da66d079f77bb9db87c786d535c3a4d12cee4d5ea56aee8bb66f4ed118ce2",
        ),
        (
            PYTHON_ROOT_V2,
            "fdeb51c8e8d439c97a3e0bbc968158d6fcd8ed8b83423c28f59377ff64a4c4d0",
        ),
        (
            PYTHON_TIMESTAMP_V2,
            "a0a254b6abefbe94f7998bee7685a829bdd9aa76762fe4ff9a3e42a9ba5f3fec",
        ),
        (
            PYTHON_SNAPSHOT_V2,
            "7558b9da6209b74e43a86bf446f0cda215dcde2383dede9d9966360522661e87",
        ),
        (
            PYTHON_TARGETS_V2,
            "d456a636121f45fee404f96cf98462721d4884e543be11a4cc874d290bcb543d",
        ),
    ] {
        assert_eq!(hex::encode(Sha256::digest(bytes)), expected);
    }

    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(PYTHON_ROOT_V1);
    let root = request(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:35:00.123456789Z"),
                instant("2026-08-18T08:35:00.223456789Z"),
            ],
        )
        .expect("Python-TUF anchor starts the production verifier"),
        "2.root.json",
    );
    let terminal = request(
        root.respond(MetadataResponse::Found(PYTHON_ROOT_V2.into()))
            .expect("Python-TUF root satisfies both two-of-two thresholds"),
        "3.root.json",
    );
    let timestamp = request(
        terminal
            .respond(MetadataResponse::ConfirmedNotFound)
            .expect("explicit root-chain termination advances the transcript"),
        "timestamp.json",
    );
    let snapshot = request(
        timestamp
            .respond(MetadataResponse::Found(PYTHON_TIMESTAMP_V2.into()))
            .expect("Python-TUF timestamp authenticates"),
        "2.snapshot.json",
    );
    let targets = request(
        snapshot
            .respond(MetadataResponse::Found(PYTHON_SNAPSHOT_V2.into()))
            .expect("Python-TUF snapshot pins authenticate"),
        "2.targets.json",
    );
    let candidate = match targets
        .respond(MetadataResponse::Found(PYTHON_TARGETS_V2.into()))
        .expect("Python-TUF targets authenticate")
    {
        VerificationStep::Candidate(candidate) => candidate,
        VerificationStep::Request(_) => panic!("complete Python-TUF transcript requested more"),
    };
    assert_eq!(candidate.anchor_root().bytes(), PYTHON_ROOT_V1);
    assert_eq!(candidate.root_chain().len(), 1);
    assert_eq!(candidate.root_chain()[0].bytes(), PYTHON_ROOT_V2);
    assert_eq!(candidate.timestamp().bytes(), PYTHON_TIMESTAMP_V2);
    assert_eq!(candidate.snapshot().bytes(), PYTHON_SNAPSHOT_V2);
    assert_eq!(candidate.targets().bytes(), PYTHON_TARGETS_V2);

    let targets = profile::targets(PYTHON_TARGETS_V2).expect("retained targets profile");
    assert!(targets
        .signed
        .targets
        .contains_key("channels/stable/aarch64-apple-darwin.json"));

    let (outcome, durable) = commit_at_recorded_completion(&authorization, &anchor, candidate)
        .expect("independent Python-TUF candidate commits and reopens");
    assert_eq!(outcome, MetadataCommitOutcome::Committed { sequence: 1 });
    assert_eq!(durable.sequence(), 1);
}

#[test]
fn independent_python_tuf_thresholds_and_parent_pins_fail_closed() {
    const OLD_B: &str = "46f2d0a47d2ec6c0254b0454ea3c5395de4837392788527fd782d585d923e267";
    const NEW_B: &str = "a8007901c02a96e7414dcb1e7694c7462913ff6e2526c12f9875dc3f6a508075";

    fn without_signature(bytes: &[u8], keyid: &str) -> Box<[u8]> {
        let mut value: serde_json::Value = serde_json::from_slice(bytes).expect("fixture JSON");
        value["signatures"]
            .as_array_mut()
            .expect("signature array")
            .retain(|signature| signature["keyid"] != keyid);
        serde_json::to_vec(&value)
            .expect("mutated fixture JSON")
            .into_boxed_slice()
    }

    let root_one: serde_json::Value =
        serde_json::from_slice(PYTHON_ROOT_V1).expect("root-one JSON");
    let root_two: serde_json::Value =
        serde_json::from_slice(PYTHON_ROOT_V2).expect("root-two JSON");
    for role in ["root", "snapshot", "targets", "timestamp"] {
        assert_eq!(root_one["signed"]["roles"][role]["threshold"], 2);
        assert_eq!(
            root_one["signed"]["roles"][role]["keyids"]
                .as_array()
                .expect("old role keys")
                .len(),
            2
        );
        assert_eq!(root_two["signed"]["roles"][role]["threshold"], 2);
        assert_eq!(
            root_two["signed"]["roles"][role]["keyids"]
                .as_array()
                .expect("new role keys")
                .len(),
            2
        );
    }
    assert_eq!(
        root_two["signatures"]
            .as_array()
            .expect("dual-threshold signatures")
            .len(),
        4
    );

    for missing in [OLD_B, NEW_B] {
        let (_temp, authorization) = authorization();
        let anchor = EmbeddedTrustRoot::from_compiled(PYTHON_ROOT_V1);
        let root = request(
            begin_from_anchor_for_test(
                &authorization,
                &anchor,
                [
                    instant("2026-08-18T08:36:00Z"),
                    instant("2026-08-18T08:36:01Z"),
                ],
            )
            .expect("Python-TUF anchor starts"),
            "2.root.json",
        );
        assert!(matches!(
            root.respond(MetadataResponse::Found(without_signature(
                PYTHON_ROOT_V2,
                missing
            ))),
            Err(TufVerifierError::AuthenticationFailed)
        ));
    }

    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(PYTHON_ROOT_V1);
    let timestamp_request = || {
        let root = request(
            begin_from_anchor_for_test(
                &authorization,
                &anchor,
                [
                    instant("2026-08-18T08:37:00Z"),
                    instant("2026-08-18T08:37:01Z"),
                ],
            )
            .expect("Python-TUF anchor starts"),
            "2.root.json",
        );
        let terminal = request(
            root.respond(MetadataResponse::Found(PYTHON_ROOT_V2.into()))
                .expect("complete root threshold authenticates"),
            "3.root.json",
        );
        request(
            terminal
                .respond(MetadataResponse::ConfirmedNotFound)
                .expect("root chain terminates"),
            "timestamp.json",
        )
    };

    assert!(matches!(
        timestamp_request().respond(MetadataResponse::Found(without_signature(
            PYTHON_TIMESTAMP_V2,
            NEW_B
        ))),
        Err(TufVerifierError::AuthenticationFailed)
    ));
    let snapshot_request = request(
        timestamp_request()
            .respond(MetadataResponse::Found(PYTHON_TIMESTAMP_V2.into()))
            .expect("complete timestamp threshold authenticates"),
        "2.snapshot.json",
    );
    assert!(matches!(
        snapshot_request.respond(MetadataResponse::Found(without_signature(
            PYTHON_SNAPSHOT_V2,
            NEW_B
        ))),
        Err(TufVerifierError::AuthenticationFailed)
    ));
    let snapshot_request = request(
        timestamp_request()
            .respond(MetadataResponse::Found(PYTHON_TIMESTAMP_V2.into()))
            .expect("complete timestamp threshold authenticates"),
        "2.snapshot.json",
    );
    let targets_request = request(
        snapshot_request
            .respond(MetadataResponse::Found(PYTHON_SNAPSHOT_V2.into()))
            .expect("complete snapshot threshold authenticates"),
        "2.targets.json",
    );
    assert!(matches!(
        targets_request.respond(MetadataResponse::Found(without_signature(
            PYTHON_TARGETS_V2,
            NEW_B
        ))),
        Err(TufVerifierError::AuthenticationFailed)
    ));

    let timestamp = profile::timestamp(PYTHON_TIMESTAMP_V2).expect("timestamp profile");
    let snapshot_pin = timestamp
        .signed
        .meta
        .get("snapshot.json")
        .expect("snapshot pin");
    assert_eq!(snapshot_pin.length, Some(PYTHON_SNAPSHOT_V2.len() as u64));
    assert_eq!(
        snapshot_pin.hashes.as_ref().expect("snapshot hashes")["sha256"],
        hex::encode(Sha256::digest(PYTHON_SNAPSHOT_V2))
    );
    let snapshot = profile::snapshot(PYTHON_SNAPSHOT_V2).expect("snapshot profile");
    let targets_pin = snapshot
        .signed
        .meta
        .get("targets.json")
        .expect("targets pin");
    assert_eq!(targets_pin.length, Some(PYTHON_TARGETS_V2.len() as u64));
    assert_eq!(
        targets_pin.hashes.as_ref().expect("targets hashes")["sha256"],
        hex::encode(Sha256::digest(PYTHON_TARGETS_V2))
    );
}
