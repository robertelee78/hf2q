use super::{response, ExpectedRequest, FailingReader, ScriptedExecutor};
use crate::distribution::install_state::metadata::{
    read_selected, MetadataCommitOutcome, MetadataStateAuthorization,
};
use crate::distribution::update_auth::{
    begin_metadata_update, EmbeddedTrustRoot, MetadataRequestKind,
};
use crate::distribution::update_transport::http::build_request;
use crate::distribution::update_transport::http::HttpResponse;
use crate::distribution::update_transport::metadata::{
    read_bounded_metadata_for_test, refresh_metadata,
};
use crate::distribution::update_transport::origin::{
    metadata_name_allowed_for_test, pages_metadata_url,
};
use crate::distribution::update_transport::UpdateTransportError;
use tempfile::TempDir;

const ROOT: &[u8] = include_bytes!("../../update_auth/testdata/tuf-v1/root-v1.json");
const TIMESTAMP: &[u8] =
    include_bytes!("../../update_auth/testdata/tuf-v1/timestamp-v2-normalized.json");
const SNAPSHOT: &[u8] = include_bytes!("../../update_auth/testdata/tuf-v1/snapshot-v2.json");
const TARGETS: &[u8] = include_bytes!("../../update_auth/testdata/tuf-v1/targets-v2.json");

const PYTHON_ROOT_V1: &[u8] =
    include_bytes!("../../update_auth/testdata/python-tuf-v1/1.root.json");
const PYTHON_ROOT_V2: &[u8] =
    include_bytes!("../../update_auth/testdata/python-tuf-v1/2.root.json");
const PYTHON_TIMESTAMP: &[u8] =
    include_bytes!("../../update_auth/testdata/python-tuf-v1/timestamp.json");
const PYTHON_SNAPSHOT: &[u8] =
    include_bytes!("../../update_auth/testdata/python-tuf-v1/2.snapshot.json");
const PYTHON_TARGETS: &[u8] =
    include_bytes!("../../update_auth/testdata/python-tuf-v1/2.targets.json");

const METADATA_BASE: &str = "https://robertelee78.github.io/hf2q/updates/stable/metadata/";

fn make_authorization() -> (TempDir, MetadataStateAuthorization) {
    let temp = TempDir::new().expect("tempdir");
    let root = temp
        .path()
        .canonicalize()
        .expect("canonical tempdir")
        .join("state");
    let authorization =
        MetadataStateAuthorization::for_test_path(&root, "11111111-1111-4111-8111-111111111111");
    (temp, authorization)
}

fn exact_response(bytes: &[u8]) -> HttpResponse {
    let mut response = response(200, bytes);
    response.content_length = Some(bytes.len() as u64);
    response
}

fn unversioned_executor() -> ScriptedExecutor {
    ScriptedExecutor::new([
        ExpectedRequest {
            url: format!("{METADATA_BASE}2.root.json"),
            large: false,
            response: response(404, b"not found"),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}timestamp.json"),
            large: false,
            response: exact_response(TIMESTAMP),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}snapshot.json"),
            large: false,
            response: exact_response(SNAPSHOT),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}targets.json"),
            large: false,
            response: exact_response(TARGETS),
        },
    ])
}

#[test]
fn metadata_session_fetches_exact_unversioned_roles_and_commits() {
    let (_temp, authorization) = make_authorization();
    let anchor = EmbeddedTrustRoot::for_test(ROOT);
    let pending = std::path::Path::new(authorization.state_root())
        .join("update/metadata/generations/.pending-00000000000000000001");
    std::fs::create_dir_all(&pending).expect("empty exact pending prefix");
    let generations = pending.parent().expect("generations");
    let metadata = generations.parent().expect("metadata");
    for directory in [metadata, generations, pending.as_path()] {
        std::fs::set_permissions(
            directory,
            std::os::unix::fs::PermissionsExt::from_mode(0o700),
        )
        .expect("pending prefix mode");
    }
    assert!(read_selected(&authorization).is_err());

    let executor = unversioned_executor();

    assert_eq!(
        refresh_metadata(&executor, &authorization, &anchor).expect("metadata refresh"),
        MetadataCommitOutcome::Committed { sequence: 1 }
    );
    executor.assert_finished();
    let selected = read_selected(&authorization)
        .expect("selected read")
        .expect("selected generation");
    assert_eq!(selected.sequence(), 1);
    assert_eq!(selected.anchor_root(), ROOT);
    assert_eq!(selected.timestamp(), TIMESTAMP);
    assert_eq!(selected.snapshot(), SNAPSHOT);
    assert_eq!(selected.targets(), TARGETS);
    assert!(!pending.exists(), "fresh session discarded exact residue");

    let executor = unversioned_executor();
    assert_eq!(
        refresh_metadata(&executor, &authorization, &anchor).expect("selected-floor refresh"),
        MetadataCommitOutcome::Committed { sequence: 2 }
    );
    executor.assert_finished();
    assert_eq!(
        read_selected(&authorization)
            .expect("selected read")
            .expect("selected generation")
            .sequence(),
        2
    );
}

#[test]
fn metadata_session_rotates_root_and_uses_consistent_snapshot_names() {
    let (_temp, authorization) = make_authorization();
    let anchor = EmbeddedTrustRoot::for_test(PYTHON_ROOT_V1);
    let executor = ScriptedExecutor::new([
        ExpectedRequest {
            url: format!("{METADATA_BASE}2.root.json"),
            large: false,
            response: exact_response(PYTHON_ROOT_V2),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}3.root.json"),
            large: false,
            response: response(404, b"not found"),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}timestamp.json"),
            large: false,
            response: exact_response(PYTHON_TIMESTAMP),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}2.snapshot.json"),
            large: false,
            response: exact_response(PYTHON_SNAPSHOT),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}2.targets.json"),
            large: false,
            response: exact_response(PYTHON_TARGETS),
        },
    ]);

    assert_eq!(
        refresh_metadata(&executor, &authorization, &anchor).expect("metadata refresh"),
        MetadataCommitOutcome::Committed { sequence: 1 }
    );
    executor.assert_finished();
    let selected = read_selected(&authorization)
        .expect("selected read")
        .expect("selected generation");
    assert_eq!(selected.root_chain().len(), 1);
    assert_eq!(selected.root_chain()[0].as_ref(), PYTHON_ROOT_V2);
}

#[test]
fn only_the_next_root_may_map_an_exact_404_to_not_found() {
    let (_temp, authorization) = make_authorization();
    let anchor = EmbeddedTrustRoot::for_test(ROOT);
    let executor = ScriptedExecutor::new([
        ExpectedRequest {
            url: format!("{METADATA_BASE}2.root.json"),
            large: false,
            response: response(404, b"not found"),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}timestamp.json"),
            large: false,
            response: response(404, b"not found"),
        },
    ]);
    assert!(matches!(
        refresh_metadata(&executor, &authorization, &anchor),
        Err(UpdateTransportError::Status)
    ));
    executor.assert_finished();
    assert!(read_selected(&authorization)
        .expect("selected read")
        .is_none());

    let (_temp, authorization) = make_authorization();
    let mut redirected = response(302, b"");
    redirected.location = Some(format!("{METADATA_BASE}missing"));
    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: format!("{METADATA_BASE}2.root.json"),
        large: false,
        response: redirected,
    }]);
    assert!(matches!(
        refresh_metadata(&executor, &authorization, &anchor),
        Err(UpdateTransportError::Status)
    ));

    for status in [201, 204, 206, 304, 401, 403, 429, 500, 503] {
        let (_temp, authorization) = make_authorization();
        let executor = ScriptedExecutor::new([ExpectedRequest {
            url: format!("{METADATA_BASE}2.root.json"),
            large: false,
            response: response(status, b"unexpected status"),
        }]);
        assert!(matches!(
            refresh_metadata(&executor, &authorization, &anchor),
            Err(UpdateTransportError::Status)
        ));
        executor.assert_finished();
    }
}

#[test]
fn metadata_bodies_are_nonempty_bounded_exact_and_untransformed() {
    assert_eq!(
        read_bounded_metadata_for_test(response(200, b"1234"), 4).expect("exact cap"),
        b"1234"
    );
    for body in [&b""[..], &b"12345"[..]] {
        assert!(matches!(
            read_bounded_metadata_for_test(response(200, body), 4),
            Err(UpdateTransportError::Length)
        ));
    }

    let mut mismatch = response(200, b"1234");
    mismatch.content_length = Some(3);
    assert!(matches!(
        read_bounded_metadata_for_test(mismatch, 4),
        Err(UpdateTransportError::Length)
    ));

    let mut oversized_header = response(200, b"1234");
    oversized_header.content_length = Some(5);
    assert!(matches!(
        read_bounded_metadata_for_test(oversized_header, 4),
        Err(UpdateTransportError::Length)
    ));

    let mut zero_header = response(200, b"1234");
    zero_header.content_length = Some(0);
    assert!(matches!(
        read_bounded_metadata_for_test(zero_header, 4),
        Err(UpdateTransportError::Length)
    ));

    let mut transformed = response(200, b"1234");
    transformed.content_encoding = Some("gzip".to_owned());
    assert!(matches!(
        read_bounded_metadata_for_test(transformed, 4),
        Err(UpdateTransportError::ContentEncoding)
    ));

    let failing = HttpResponse {
        status: 200,
        content_length: None,
        content_encoding: None,
        location: None,
        body: Box::new(FailingReader),
    };
    assert!(matches!(
        read_bounded_metadata_for_test(failing, 4),
        Err(UpdateTransportError::BodyRead)
    ));
}

#[test]
fn unauthenticated_network_metadata_never_reaches_the_journal() {
    let (_temp, authorization) = make_authorization();
    let anchor = EmbeddedTrustRoot::for_test(ROOT);
    let mut mutated_timestamp = TIMESTAMP.to_vec();
    let signature_prefix = b"\"sig\": \"";
    let signature = mutated_timestamp
        .windows(signature_prefix.len())
        .position(|window| window == signature_prefix)
        .expect("timestamp signature")
        + signature_prefix.len();
    mutated_timestamp[signature] = if mutated_timestamp[signature] == b'0' {
        b'1'
    } else {
        b'0'
    };
    let executor = ScriptedExecutor::new([
        ExpectedRequest {
            url: format!("{METADATA_BASE}2.root.json"),
            large: false,
            response: response(404, b"not found"),
        },
        ExpectedRequest {
            url: format!("{METADATA_BASE}timestamp.json"),
            large: false,
            response: exact_response(&mutated_timestamp),
        },
    ]);
    assert!(matches!(
        refresh_metadata(&executor, &authorization, &anchor),
        Err(UpdateTransportError::Authentication(_))
    ));
    executor.assert_finished();
    assert!(read_selected(&authorization)
        .expect("selected read")
        .is_none());
}

#[test]
fn metadata_names_are_a_closed_single_component_grammar() {
    for (kind, accepted) in [
        (MetadataRequestKind::Root, "1.root.json"),
        (MetadataRequestKind::Timestamp, "timestamp.json"),
        (MetadataRequestKind::Snapshot, "snapshot.json"),
        (MetadataRequestKind::Snapshot, "2.snapshot.json"),
        (MetadataRequestKind::Targets, "targets.json"),
        (MetadataRequestKind::Targets, "2.targets.json"),
    ] {
        assert!(metadata_name_allowed_for_test(kind, accepted));
    }

    for (kind, rejected) in [
        (MetadataRequestKind::Root, "0.root.json"),
        (MetadataRequestKind::Root, "01.root.json"),
        (MetadataRequestKind::Root, "root.json"),
        (MetadataRequestKind::Timestamp, "1.timestamp.json"),
        (MetadataRequestKind::Snapshot, "0.snapshot.json"),
        (MetadataRequestKind::Targets, "01.targets.json"),
        (MetadataRequestKind::Targets, "../targets.json"),
        (MetadataRequestKind::Targets, "nested/targets.json"),
        (MetadataRequestKind::Targets, "targets.json?query"),
    ] {
        assert!(
            !metadata_name_allowed_for_test(kind, rejected),
            "accepted hostile metadata name: {rejected}"
        );
    }
}

#[test]
fn metadata_session_debug_exposes_only_the_request_view() {
    let (_temp, authorization) = make_authorization();
    let anchor = EmbeddedTrustRoot::for_test(ROOT);
    let session = begin_metadata_update(&authorization, &anchor).expect("metadata session");
    let debug = format!("{session:?}");
    assert!(debug.contains("2.root.json"));
    assert!(debug.contains("maximum_bytes"));
    assert!(!debug.contains("signed"));
    assert!(!debug.contains("keys"));
    assert!(!debug.contains("signatures"));

    let url = pages_metadata_url(session.request()).expect("metadata URL");
    assert_eq!(url.url().as_str(), format!("{METADATA_BASE}2.root.json"));
    let client = reqwest::blocking::Client::new();
    let request = build_request(&client, &url).expect("metadata request");
    assert_eq!(
        request.headers().get(reqwest::header::ACCEPT_ENCODING),
        Some(&reqwest::header::HeaderValue::from_static("identity"))
    );
    assert_eq!(
        request.headers().get(reqwest::header::CACHE_CONTROL),
        Some(&reqwest::header::HeaderValue::from_static("no-cache"))
    );
    assert!(!request.headers().contains_key(reqwest::header::REFERER));
    assert!(!request.headers().contains_key(reqwest::header::RANGE));
}
