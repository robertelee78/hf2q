use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{self, Cursor, Read};

use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::fetch::{fetch_direct, fetch_release, stream_archive};
use super::http::{build_request, parse_headers, HttpExecutor, HttpResponse};
use super::origin::{pages_pointer_url, release_asset_url, release_redirect, TargetFetchSpec};
use super::UpdateTransportError;
use crate::distribution::install_state::metadata::MetadataStateAuthorization;
use crate::distribution::install_state::{
    create_ephemeral_artifact_stage, ExplicitRootAuthorization,
};
use crate::distribution::schema::{LogicalTargetName, ReleaseVersion, TargetTriple, UpdateChannel};
use crate::distribution::update_auth::AuthenticatedTargetDescriptor;

struct ExpectedRequest {
    url: String,
    large: bool,
    response: HttpResponse,
}

#[derive(Default)]
struct ScriptedExecutor {
    requests: RefCell<VecDeque<ExpectedRequest>>,
}

struct FailingReader;

impl Read for FailingReader {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "scripted body failure",
        ))
    }
}

impl ScriptedExecutor {
    fn new(requests: impl IntoIterator<Item = ExpectedRequest>) -> Self {
        Self {
            requests: RefCell::new(requests.into_iter().collect()),
        }
    }

    fn assert_finished(&self) {
        assert!(self.requests.borrow().is_empty(), "unconsumed request");
    }
}

impl HttpExecutor for ScriptedExecutor {
    fn execute(
        &self,
        request: &super::origin::OriginLockedUrl,
        large: bool,
    ) -> Result<HttpResponse, UpdateTransportError> {
        let expected = self
            .requests
            .borrow_mut()
            .pop_front()
            .expect("unexpected request");
        assert_eq!(request.url().as_str(), expected.url);
        assert_eq!(large, expected.large);
        Ok(expected.response)
    }
}

fn response(status: u16, body: &[u8]) -> HttpResponse {
    HttpResponse {
        status,
        content_length: None,
        content_encoding: None,
        location: None,
        body: Box::new(Cursor::new(body.to_vec())),
    }
}

fn pointer_descriptor(bytes: &[u8]) -> AuthenticatedTargetDescriptor {
    AuthenticatedTargetDescriptor::for_test(
        LogicalTargetName::channel_pointer(UpdateChannel::Stable, TargetTriple::Aarch64AppleDarwin),
        bytes,
    )
}

fn manifest_descriptor(version: &ReleaseVersion, bytes: &[u8]) -> AuthenticatedTargetDescriptor {
    AuthenticatedTargetDescriptor::for_test(
        LogicalTargetName::release_manifest(version, TargetTriple::Aarch64AppleDarwin),
        bytes,
    )
}

fn archive_descriptor(version: &ReleaseVersion, bytes: &[u8]) -> AuthenticatedTargetDescriptor {
    AuthenticatedTargetDescriptor::for_test(
        LogicalTargetName::release_archive(version, TargetTriple::Aarch64AppleDarwin),
        bytes,
    )
}

#[test]
fn typed_routes_are_exact_and_release_assets_are_flat() {
    let pointer = pointer_descriptor(b"pointer");
    let pointer_spec = TargetFetchSpec::from_descriptor(&pointer).expect("pointer spec");
    let pointer_url = pages_pointer_url(&pointer_spec).expect("pointer URL");
    let pointer_sha = hex::encode(Sha256::digest(b"pointer"));
    assert_eq!(
        pointer_url.url().as_str(),
        format!(
            "https://robertelee78.github.io/hf2q/updates/stable/targets/channels/stable/{pointer_sha}.aarch64-apple-darwin.json"
        )
    );

    let version = ReleaseVersion::parse_stable("version", "1.2.3".to_owned()).expect("version");
    let manifest = manifest_descriptor(&version, b"manifest");
    let manifest_spec = TargetFetchSpec::from_descriptor(&manifest).expect("manifest spec");
    let release_url = release_asset_url(&version, &manifest_spec).expect("release URL");
    let manifest_sha = hex::encode(Sha256::digest(b"manifest"));
    assert_eq!(
        release_url.url().as_str(),
        format!(
            "https://github.com/robertelee78/hf2q/releases/download/v1.2.3/{manifest_sha}.release-manifest.json"
        )
    );
    assert!(!release_url.url().path().contains("releases/v1.2.3/aarch64"));
}

#[test]
fn request_headers_are_explicit_and_response_headers_are_singleton_bounded() {
    let client = reqwest::blocking::Client::new();
    let pointer = pointer_descriptor(b"pointer");
    let pointer_spec = TargetFetchSpec::from_descriptor(&pointer).expect("pointer spec");
    let pointer_url = pages_pointer_url(&pointer_spec).expect("pointer URL");
    let pointer_request = build_request(&client, &pointer_url).expect("pointer request");
    assert_eq!(
        pointer_request
            .headers()
            .get(reqwest::header::ACCEPT_ENCODING),
        Some(&reqwest::header::HeaderValue::from_static("identity"))
    );
    assert_eq!(
        pointer_request
            .headers()
            .get(reqwest::header::CACHE_CONTROL),
        Some(&reqwest::header::HeaderValue::from_static("no-cache"))
    );
    assert!(!pointer_request
        .headers()
        .contains_key(reqwest::header::REFERER));
    assert!(!pointer_request
        .headers()
        .contains_key(reqwest::header::RANGE));

    let version = ReleaseVersion::parse_stable("version", "1.2.3".to_owned()).expect("version");
    let manifest = manifest_descriptor(&version, b"manifest");
    let manifest_spec = TargetFetchSpec::from_descriptor(&manifest).expect("manifest spec");
    let release_url = release_asset_url(&version, &manifest_spec).expect("release URL");
    let release_request = build_request(&client, &release_url).expect("release request");
    assert!(!release_request
        .headers()
        .contains_key(reqwest::header::CACHE_CONTROL));

    let mut duplicated = reqwest::header::HeaderMap::new();
    duplicated.append(
        reqwest::header::CONTENT_LENGTH,
        reqwest::header::HeaderValue::from_static("1"),
    );
    duplicated.append(
        reqwest::header::CONTENT_LENGTH,
        reqwest::header::HeaderValue::from_static("1"),
    );
    assert!(matches!(
        parse_headers(&duplicated),
        Err(UpdateTransportError::Headers)
    ));

    let mut oversized = reqwest::header::HeaderMap::new();
    oversized.insert(
        reqwest::header::LOCATION,
        reqwest::header::HeaderValue::from_bytes(&vec![b'a'; 16 * 1024 + 1])
            .expect("large visible header"),
    );
    assert!(matches!(
        parse_headers(&oversized),
        Err(UpdateTransportError::Headers)
    ));
}

#[test]
fn redirect_policy_accepts_only_the_exact_cdn_shape() {
    let accepted = release_redirect(
        "https://release-assets.githubusercontent.com/github-production-release-asset/1/object?sig=secret",
    )
    .expect("accepted redirect");
    let debug = format!("{accepted:?}");
    assert!(!debug.contains("secret"));
    assert!(!debug.contains("github-production-release-asset"));
    release_redirect("https://release-assets.githubusercontent.com:443/file")
        .expect("explicit default TLS port");

    for rejected in [
        "http://release-assets.githubusercontent.com/file",
        "https://release-assets.githubusercontent.com.evil.example/file",
        "https://evil-release-assets.githubusercontent.com/file",
        "https://user@release-assets.githubusercontent.com/file",
        "https://@release-assets.githubusercontent.com/file",
        "https://release-assets.githubusercontent.com:444/file",
        "https://release-assets.githubusercontent.com/file#fragment",
        "https://127.0.0.1/file",
        "/relative",
    ] {
        assert!(
            matches!(
                release_redirect(rejected),
                Err(UpdateTransportError::OriginPolicy)
            ),
            "accepted hostile redirect: {rejected}"
        );
    }
}

#[test]
fn exact_body_succeeds_without_content_length_and_rejects_mutation() {
    let bytes = b"pointer";
    let descriptor = pointer_descriptor(bytes);
    let spec = TargetFetchSpec::from_descriptor(&descriptor).expect("spec");
    let request = pages_pointer_url(&spec).expect("URL");
    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: request.url().to_string(),
        large: false,
        response: response(200, bytes),
    }]);
    assert_eq!(
        fetch_direct(&executor, &request, &spec, false).expect("fetch"),
        bytes
    );
    executor.assert_finished();

    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: request.url().to_string(),
        large: false,
        response: response(200, b"pointex"),
    }]);
    assert!(matches!(
        fetch_direct(&executor, &request, &spec, false),
        Err(UpdateTransportError::Digest)
    ));
}

#[test]
fn headers_status_and_length_fail_closed() {
    let bytes = b"pointer";
    let descriptor = pointer_descriptor(bytes);
    let spec = TargetFetchSpec::from_descriptor(&descriptor).expect("spec");
    let request = pages_pointer_url(&spec).expect("URL");

    for mut bad in [
        HttpResponse {
            status: 302,
            content_length: None,
            content_encoding: None,
            location: Some("https://example.com".to_owned()),
            body: Box::new(Cursor::new(Vec::new())),
        },
        HttpResponse {
            status: 200,
            content_length: Some(bytes.len() as u64 + 1),
            content_encoding: None,
            location: None,
            body: Box::new(Cursor::new(bytes.to_vec())),
        },
        HttpResponse {
            status: 200,
            content_length: None,
            content_encoding: Some("gzip".to_owned()),
            location: None,
            body: Box::new(Cursor::new(bytes.to_vec())),
        },
        HttpResponse {
            status: 206,
            content_length: None,
            content_encoding: None,
            location: None,
            body: Box::new(Cursor::new(bytes.to_vec())),
        },
        HttpResponse {
            status: 429,
            content_length: None,
            content_encoding: None,
            location: None,
            body: Box::new(Cursor::new(bytes.to_vec())),
        },
        HttpResponse {
            status: 500,
            content_length: None,
            content_encoding: None,
            location: None,
            body: Box::new(Cursor::new(bytes.to_vec())),
        },
    ] {
        let executor = ScriptedExecutor::new([ExpectedRequest {
            url: request.url().to_string(),
            large: false,
            response: HttpResponse {
                status: bad.status,
                content_length: bad.content_length,
                content_encoding: bad.content_encoding.take(),
                location: bad.location.take(),
                body: bad.body,
            },
        }]);
        assert!(fetch_direct(&executor, &request, &spec, false).is_err());
    }
}

#[test]
fn short_and_failed_bodies_never_produce_authenticated_bytes() {
    let bytes = b"pointer";
    let descriptor = pointer_descriptor(bytes);
    let spec = TargetFetchSpec::from_descriptor(&descriptor).expect("spec");
    let request = pages_pointer_url(&spec).expect("URL");

    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: request.url().to_string(),
        large: false,
        response: response(200, b"short"),
    }]);
    assert!(matches!(
        fetch_direct(&executor, &request, &spec, false),
        Err(UpdateTransportError::Length)
    ));

    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: request.url().to_string(),
        large: false,
        response: HttpResponse {
            status: 200,
            content_length: None,
            content_encoding: None,
            location: None,
            body: Box::new(FailingReader),
        },
    }]);
    assert!(matches!(
        fetch_direct(&executor, &request, &spec, false),
        Err(UpdateTransportError::BodyRead)
    ));
}

#[test]
fn release_allows_one_302_and_rejects_a_second() {
    let version = ReleaseVersion::parse_stable("version", "1.2.3".to_owned()).expect("version");
    let descriptor = manifest_descriptor(&version, b"manifest");
    let spec = TargetFetchSpec::from_descriptor(&descriptor).expect("spec");
    let initial = release_asset_url(&version, &spec).expect("initial");
    let cdn = "https://release-assets.githubusercontent.com/asset/object?sig=secret";
    let mut redirect = response(302, b"");
    redirect.location = Some(cdn.to_owned());
    let executor = ScriptedExecutor::new([
        ExpectedRequest {
            url: initial.url().to_string(),
            large: false,
            response: redirect,
        },
        ExpectedRequest {
            url: cdn.to_owned(),
            large: false,
            response: response(200, b"manifest"),
        },
    ]);
    assert_eq!(
        fetch_release(&executor, &initial, &spec, false).expect("release fetch"),
        b"manifest"
    );
    executor.assert_finished();

    let mut first = response(302, b"");
    first.location = Some(cdn.to_owned());
    let mut second = response(302, b"");
    second.location = Some(cdn.to_owned());
    let executor = ScriptedExecutor::new([
        ExpectedRequest {
            url: initial.url().to_string(),
            large: false,
            response: first,
        },
        ExpectedRequest {
            url: cdn.to_owned(),
            large: false,
            response: second,
        },
    ]);
    assert!(matches!(
        fetch_release(&executor, &initial, &spec, false),
        Err(UpdateTransportError::Status)
    ));

    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: initial.url().to_string(),
        large: false,
        response: response(302, b""),
    }]);
    assert!(matches!(
        fetch_release(&executor, &initial, &spec, false),
        Err(UpdateTransportError::Headers)
    ));

    let executor = ScriptedExecutor::new([ExpectedRequest {
        url: initial.url().to_string(),
        large: false,
        response: response(200, b"manifest"),
    }]);
    assert_eq!(
        fetch_release(&executor, &initial, &spec, false).expect("direct release fetch"),
        b"manifest"
    );
    executor.assert_finished();
}

#[test]
fn archive_stream_is_bounded_and_finishes_the_same_fd() {
    let bytes = b"archive bytes streamed in chunks";
    let version = ReleaseVersion::parse_stable("version", "1.2.3".to_owned()).expect("version");
    let descriptor = archive_descriptor(&version, bytes);
    let spec = TargetFetchSpec::from_descriptor(&descriptor).expect("spec");
    let temp = TempDir::new().expect("tempdir");
    let state = temp
        .path()
        .canonicalize()
        .expect("canonical tempdir")
        .join("state");
    std::fs::create_dir(&state).expect("state root");
    std::fs::set_permissions(&state, std::os::unix::fs::PermissionsExt::from_mode(0o700))
        .expect("permissions");
    let authorization = MetadataStateAuthorization::for_test(
        ExplicitRootAuthorization::new(&state).expect("root auth"),
        "11111111-1111-4111-8111-111111111111",
    );
    let stage = create_ephemeral_artifact_stage(&authorization, bytes.len() as u64)
        .expect("artifact stage");
    let mut verified =
        stream_archive(response(200, bytes), &spec, stage).expect("verified archive");
    let mut reread = Vec::new();
    verified.read_to_end(&mut reread).expect("reread");
    assert_eq!(reread, bytes);

    let stage = create_ephemeral_artifact_stage(&authorization, bytes.len() as u64)
        .expect("second artifact stage");
    let mut overlong = bytes.to_vec();
    overlong.push(b'!');
    assert!(matches!(
        stream_archive(response(200, &overlong), &spec, stage),
        Err(UpdateTransportError::Length)
    ));

    let stage = create_ephemeral_artifact_stage(&authorization, bytes.len() as u64)
        .expect("short artifact stage");
    assert!(matches!(
        stream_archive(response(200, &bytes[..bytes.len() - 1]), &spec, stage),
        Err(UpdateTransportError::Length)
    ));

    let stage = create_ephemeral_artifact_stage(&authorization, bytes.len() as u64)
        .expect("mutated artifact stage");
    let mut mutated = bytes.to_vec();
    mutated[0] ^= 1;
    assert!(matches!(
        stream_archive(response(200, &mutated), &spec, stage),
        Err(UpdateTransportError::Stage(
            crate::distribution::install_state::ArtifactStageError::Integrity
        ))
    ));

    let stage = create_ephemeral_artifact_stage(&authorization, bytes.len() as u64)
        .expect("failed-body artifact stage");
    let failed = HttpResponse {
        status: 200,
        content_length: None,
        content_encoding: None,
        location: None,
        body: Box::new(FailingReader),
    };
    assert!(matches!(
        stream_archive(failed, &spec, stage),
        Err(UpdateTransportError::BodyRead)
    ));
}
