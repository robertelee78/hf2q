//! Reachable standalone-channel update: one small release record, one native
//! executable, and the same local publication primitive used by install.

use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use reqwest::blocking::{Client, Response};
use reqwest::header::{ACCEPT_ENCODING, CACHE_CONTROL, CONTENT_ENCODING, CONTENT_LENGTH};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    publish_verified_candidate, verify_running_installation, CandidateExpectation, StandaloneError,
};

const STABLE_RECORD_URL: &str = "https://hf2q.us/releases/stable-aarch64-apple-darwin.json";
const GITHUB_RELEASE_BASE: &str = "https://github.com/robertelee78/hf2q/releases/download";
const ASSET_NAME: &str = "hf2q-aarch64-apple-darwin";
const RECORD_KIND: &str = "hf2q.standalone-release";
const RECORD_SCHEMA: u32 = 1;
const RECORD_TARGET: &str = "aarch64-apple-darwin";
const MAX_RECORD_BYTES: usize = 4 * 1024;
const IO_BUFFER_BYTES: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const RECORD_TIMEOUT: Duration = Duration::from_secs(60);
const ASSET_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const MAX_SIGNING_INFO_BYTES: usize = 64 * 1024;
const MAX_VERSION_OUTPUT_BYTES: usize = 256;
const MAX_ARCHITECTURE_OUTPUT_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StableReleaseV1 {
    kind: String,
    schema_version: u32,
    package: String,
    channel: String,
    target: String,
    version: String,
    size: u64,
    sha256: String,
}

struct ValidatedRelease {
    version: semver::Version,
    version_text: String,
    expectation: CandidateExpectation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UpdateOutcome {
    Current { version: String },
    Available { current: String, latest: String },
    Updated { previous: String, current: String },
}

pub(crate) fn run_update(
    executable: &Path,
    check_only: bool,
) -> Result<UpdateOutcome, StandaloneError> {
    let executable = fs::canonicalize(executable)
        .map_err(|error| StandaloneError::io("canonicalize running executable", error))?;
    let install_directory = verify_running_installation(&executable)?;

    let release = fetch_stable_release()?;
    let current = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .map_err(|_| StandaloneError::ReleaseRecord("running version is not stable SemVer"))?;
    if release.version <= current {
        return Ok(UpdateOutcome::Current {
            version: current.to_string(),
        });
    }
    if check_only {
        return Ok(UpdateOutcome::Available {
            current: current.to_string(),
            latest: release.version_text,
        });
    }

    let mut candidate = tempfile::Builder::new()
        .prefix("hf2q-standalone-download.")
        .tempfile()
        .map_err(|error| StandaloneError::io("create update download", error))?;
    download_release_asset(&release, candidate.as_file_mut())?;
    candidate
        .as_file()
        .set_permissions(fs::Permissions::from_mode(0o555))
        .map_err(|error| StandaloneError::io("set downloaded executable mode", error))?;
    super::sync_file(candidate.as_file(), "sync downloaded executable")?;
    verify_apple_release(&executable, candidate.path(), &release.version_text)?;
    publish_verified_candidate(&install_directory, candidate.path(), &release.expectation)?;
    Ok(UpdateOutcome::Updated {
        previous: current.to_string(),
        current: release.version_text,
    })
}

fn fetch_stable_release() -> Result<ValidatedRelease, StandaloneError> {
    let client = client(RECORD_TIMEOUT, reqwest::redirect::Policy::none())?;
    let response = client
        .get(STABLE_RECORD_URL)
        .header(ACCEPT_ENCODING, "identity")
        .header(CACHE_CONTROL, "no-cache")
        .send()
        .map_err(network_error)?;
    if response.url().as_str() != STABLE_RECORD_URL {
        return Err(StandaloneError::Network(
            "stable release request changed origin".to_owned(),
        ));
    }
    require_success_response(&response, None)?;
    let bytes = read_bounded(response, MAX_RECORD_BYTES)?;
    parse_release_record(&bytes)
}

fn parse_release_record(bytes: &[u8]) -> Result<ValidatedRelease, StandaloneError> {
    if bytes.is_empty() || bytes.len() > MAX_RECORD_BYTES {
        return Err(StandaloneError::ReleaseRecord(
            "record size is outside the supported bound",
        ));
    }
    let record: StableReleaseV1 = serde_json::from_slice(bytes)
        .map_err(|_| StandaloneError::ReleaseRecord("record is not strict schema-1 JSON"))?;
    if record.kind != RECORD_KIND
        || record.schema_version != RECORD_SCHEMA
        || record.package != "hf2q"
        || record.channel != "stable"
        || record.target != RECORD_TARGET
    {
        return Err(StandaloneError::ReleaseRecord(
            "record identity does not match the hf2q stable Apple-Silicon channel",
        ));
    }
    let version = semver::Version::parse(&record.version)
        .map_err(|_| StandaloneError::ReleaseRecord("version is not stable SemVer"))?;
    if !version.pre.is_empty() || !version.build.is_empty() || version.to_string() != record.version
    {
        return Err(StandaloneError::ReleaseRecord(
            "version must be canonical stable SemVer",
        ));
    }
    let expectation = CandidateExpectation::from_hex(record.size, &record.sha256)?;
    let mut canonical = serde_json::to_vec(&record)
        .map_err(|_| StandaloneError::ReleaseRecord("record could not be canonicalized"))?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(StandaloneError::ReleaseRecord(
            "record is not the canonical managed JSON encoding",
        ));
    }
    Ok(ValidatedRelease {
        version,
        version_text: record.version,
        expectation,
    })
}

fn download_release_asset(
    release: &ValidatedRelease,
    destination: &mut fs::File,
) -> Result<(), StandaloneError> {
    let initial_url = format!(
        "{GITHUB_RELEASE_BASE}/v{}/{ASSET_NAME}",
        release.version_text
    );
    let client = client(ASSET_TIMEOUT, release_redirect_policy())?;
    let mut response = client
        .get(&initial_url)
        .header(ACCEPT_ENCODING, "identity")
        .send()
        .map_err(network_error)?;
    let final_url = response.url();
    if final_url.scheme() != "https"
        || !matches!(
            final_url.host_str(),
            Some("github.com" | "release-assets.githubusercontent.com")
        )
    {
        return Err(StandaloneError::Network(
            "release asset left the allowed GitHub origins".to_owned(),
        ));
    }
    require_success_response(&response, Some(release.expectation.size))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; IO_BUFFER_BYTES];
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|_| StandaloneError::Network("release asset read failed".to_owned()))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or(StandaloneError::ReleaseRecord(
                "release asset byte count overflowed",
            ))?;
        if total > release.expectation.size {
            return Err(StandaloneError::DigestMismatch);
        }
        hasher.update(&buffer[..read]);
        destination
            .write_all(&buffer[..read])
            .map_err(|error| StandaloneError::io("write update download", error))?;
    }
    destination
        .flush()
        .map_err(|error| StandaloneError::io("flush update download", error))?;
    let digest: [u8; 32] = hasher.finalize().into();
    if total != release.expectation.size || digest != release.expectation.sha256 {
        return Err(StandaloneError::DigestMismatch);
    }
    Ok(())
}

fn release_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        let url = attempt.url();
        if attempt.previous().len() >= 3 {
            return attempt.error("too many release-asset redirects");
        }
        if url.scheme() == "https" && url.host_str() == Some("release-assets.githubusercontent.com")
        {
            attempt.follow()
        } else {
            attempt.stop()
        }
    })
}

fn client(
    timeout: Duration,
    redirects: reqwest::redirect::Policy,
) -> Result<Client, StandaloneError> {
    Client::builder()
        .use_rustls_tls()
        .redirect(redirects)
        .retry(reqwest::retry::never())
        .referer(false)
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(timeout)
        .user_agent(concat!("hf2q/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(network_error)
}

fn require_success_response(
    response: &Response,
    expected_length: Option<u64>,
) -> Result<(), StandaloneError> {
    if !response.status().is_success() {
        return Err(StandaloneError::Network(format!(
            "server returned HTTP {}",
            response.status().as_u16()
        )));
    }
    if response.headers().contains_key(CONTENT_ENCODING) {
        return Err(StandaloneError::Network(
            "encoded responses are not accepted".to_owned(),
        ));
    }
    if let Some(expected) = expected_length {
        let actual = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok());
        if actual != Some(expected) {
            return Err(StandaloneError::DigestMismatch);
        }
    }
    Ok(())
}

fn read_bounded(response: Response, maximum: usize) -> Result<Vec<u8>, StandaloneError> {
    let mut bytes = Vec::with_capacity(maximum.min(1024));
    response
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| StandaloneError::Network("stable release read failed".to_owned()))?;
    if bytes.len() > maximum {
        return Err(StandaloneError::ReleaseRecord(
            "record size is outside the supported bound",
        ));
    }
    Ok(bytes)
}

fn network_error(error: reqwest::Error) -> StandaloneError {
    StandaloneError::Network(error.without_url().to_string())
}

#[derive(Debug, PartialEq, Eq)]
struct SigningIdentity {
    team_id: String,
    identifier: String,
}

fn verify_apple_release(
    current: &Path,
    candidate: &Path,
    version: &str,
) -> Result<(), StandaloneError> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (current, candidate, version);
        return Err(StandaloneError::Trust(
            "standalone releases require macOS Apple trust services",
        ));
    }
    #[cfg(target_os = "macos")]
    {
        verify_thin_arm64(candidate)?;
        verify_codesign(current)?;
        verify_codesign(candidate)?;
        let current_identity = signing_identity(current)?;
        let candidate_identity = signing_identity(candidate)?;
        if current_identity != candidate_identity {
            return Err(StandaloneError::Trust(
                "candidate Developer ID does not match the installed standalone binary",
            ));
        }
        command_success(
            "/usr/sbin/spctl",
            &["--assess", "--type", "execute"],
            candidate,
            "Gatekeeper did not accept the candidate",
        )?;
        let output = Command::new(candidate)
            .arg("--version")
            .env_clear()
            .env("HF2Q_NO_COMPLETION_INSTALL", "1")
            .output()
            .map_err(|error| StandaloneError::io("run candidate version check", error))?;
        if !output.status.success()
            || output.stdout.len() > MAX_VERSION_OUTPUT_BYTES
            || output.stderr.len() > MAX_VERSION_OUTPUT_BYTES
            || output.stdout != format!("hf2q {version}\n").as_bytes()
            || !output.stderr.is_empty()
        {
            return Err(StandaloneError::Trust(
                "candidate version does not match the stable release record",
            ));
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn verify_thin_arm64(path: &Path) -> Result<(), StandaloneError> {
    let output = Command::new("/usr/bin/lipo")
        .arg("-archs")
        .arg(path)
        .output()
        .map_err(|error| StandaloneError::io("inspect candidate architecture", error))?;
    if !output.status.success()
        || output.stdout.len() > MAX_ARCHITECTURE_OUTPUT_BYTES
        || output.stderr.len() > MAX_ARCHITECTURE_OUTPUT_BYTES
        || !output.stderr.is_empty()
        || parse_thin_arm64(&output.stdout).is_err()
    {
        return Err(StandaloneError::Trust(
            "candidate is not an exact thin Apple-Silicon executable",
        ));
    }
    Ok(())
}

fn parse_thin_arm64(output: &[u8]) -> Result<(), StandaloneError> {
    let text = std::str::from_utf8(output)
        .map_err(|_| StandaloneError::Trust("candidate architecture was not UTF-8"))?;
    let mut architectures = text.split_ascii_whitespace();
    if architectures.next() != Some("arm64") || architectures.next().is_some() {
        return Err(StandaloneError::Trust(
            "candidate is not an exact thin Apple-Silicon executable",
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn verify_codesign(path: &Path) -> Result<(), StandaloneError> {
    command_success(
        "/usr/bin/codesign",
        &["--verify", "--strict", "--all-architectures"],
        path,
        "Apple code-signature verification failed",
    )
}

#[cfg(target_os = "macos")]
fn command_success(
    program: &str,
    arguments: &[&str],
    path: &Path,
    failure: &'static str,
) -> Result<(), StandaloneError> {
    let status = Command::new(program)
        .args(arguments)
        .arg(path)
        .status()
        .map_err(|error| StandaloneError::io("execute Apple trust tool", error))?;
    if !status.success() {
        return Err(StandaloneError::Trust(failure));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn signing_identity(path: &Path) -> Result<SigningIdentity, StandaloneError> {
    let output = Command::new("/usr/bin/codesign")
        .args(["--display", "--verbose=4"])
        .arg(path)
        .output()
        .map_err(|error| StandaloneError::io("read Apple signing identity", error))?;
    if !output.status.success()
        || output.stderr.len() > MAX_SIGNING_INFO_BYTES
        || !output.stdout.is_empty()
    {
        return Err(StandaloneError::Trust(
            "Apple signing identity could not be read",
        ));
    }
    let text = std::str::from_utf8(&output.stderr)
        .map_err(|_| StandaloneError::Trust("Apple signing identity was not UTF-8"))?;
    let team_id = unique_signing_value(text, "TeamIdentifier=")?;
    if team_id.len() != 10
        || !team_id
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
    {
        return Err(StandaloneError::Trust(
            "Apple Developer ID team is not canonical",
        ));
    }
    let identifier = unique_signing_value(text, "Identifier=")?;
    if identifier.is_empty()
        || !identifier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
    {
        return Err(StandaloneError::Trust(
            "Apple signing identifier is not canonical",
        ));
    }
    Ok(SigningIdentity {
        team_id: team_id.to_owned(),
        identifier: identifier.to_owned(),
    })
}

#[cfg(target_os = "macos")]
fn unique_signing_value<'a>(text: &'a str, prefix: &str) -> Result<&'a str, StandaloneError> {
    let mut matches = text.lines().filter_map(|line| line.strip_prefix(prefix));
    let value = matches.next().ok_or(StandaloneError::Trust(
        "Apple signing identity is incomplete",
    ))?;
    if matches.next().is_some() {
        return Err(StandaloneError::Trust(
            "Apple signing identity contains duplicate fields",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(version: &str, size: u64, sha256: &str) -> Vec<u8> {
        format!(
            "{{\"kind\":\"hf2q.standalone-release\",\"schema_version\":1,\"package\":\"hf2q\",\"channel\":\"stable\",\"target\":\"aarch64-apple-darwin\",\"version\":\"{version}\",\"size\":{size},\"sha256\":\"{sha256}\"}}\n"
        )
        .into_bytes()
    }

    #[test]
    fn stable_release_record_is_small_canonical_and_exact() {
        let sha = "ab".repeat(32);
        let parsed = parse_release_record(&record("1.2.3", 42, &sha)).expect("valid record");
        assert_eq!(parsed.version, semver::Version::new(1, 2, 3));
        assert_eq!(parsed.expectation.size, 42);

        let mut unknown = record("1.2.3", 42, &sha);
        unknown.splice(1..1, b"\"extra\":true,".iter().copied());
        assert!(parse_release_record(&unknown).is_err());
        assert!(parse_release_record(&record("1.2.3-beta.1", 42, &sha)).is_err());
        assert!(parse_release_record(&record("01.2.3", 42, &sha)).is_err());
        let mut noncanonical = record("1.2.3", 42, &sha);
        noncanonical.insert(0, b' ');
        assert!(parse_release_record(&noncanonical).is_err());
    }

    #[test]
    fn standalone_candidate_architecture_is_exactly_thin_arm64() {
        parse_thin_arm64(b"arm64\n").expect("thin arm64");
        assert!(parse_thin_arm64(b"x86_64\n").is_err());
        assert!(parse_thin_arm64(b"arm64 x86_64\n").is_err());
        assert!(parse_thin_arm64(b"arm64e\n").is_err());
        assert!(parse_thin_arm64(b"").is_err());
        assert!(parse_thin_arm64(b"arm64\xff").is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn signing_identity_fields_are_unique_and_canonical() {
        let parsed = "Identifier=us.hf2q.cli\nTeamIdentifier=ABCDE12345\n";
        assert_eq!(
            unique_signing_value(parsed, "Identifier=").unwrap(),
            "us.hf2q.cli"
        );
        assert!(unique_signing_value(
            "Identifier=a\nIdentifier=b\nTeamIdentifier=ABCDE12345\n",
            "Identifier="
        )
        .is_err());
    }
}
