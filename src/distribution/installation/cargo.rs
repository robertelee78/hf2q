//! Cargo ownership receipts and post-delegation reconciliation.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::process::{Command, Stdio};

use serde::Deserialize;

use super::{
    detect_for_context, path_entry_exists, CargoGitSelector, CargoGitSource, CargoInstallOptions,
    CargoSource, Installation, InstallationError,
};

const CARGO_V1_RECEIPT: &str = ".crates.toml";
const CARGO_V2_RECEIPT: &str = ".crates2.json";
const MAX_RECEIPT_BYTES: u64 = 1024 * 1024;
const MAX_VERSION_OUTPUT_BYTES: usize = 256;
const CRATES_IO_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";

pub(super) struct CargoRecord {
    pub(super) version: semver::Version,
    pub(super) source: CargoSource,
    pub(super) options: CargoInstallOptions,
}

pub(super) fn inspect(
    executable: &Path,
    expected_version: &str,
) -> Result<Option<(std::path::PathBuf, CargoRecord)>, InstallationError> {
    let Some(bin_dir) = executable.parent() else {
        return Ok(None);
    };
    if bin_dir.file_name() != Some(OsStr::new("bin")) {
        return Ok(None);
    }
    let Some(root) = bin_dir.parent() else {
        return Ok(None);
    };
    let v1 = root.join(CARGO_V1_RECEIPT);
    let v2 = root.join(CARGO_V2_RECEIPT);
    let v1_exists = path_entry_exists(&v1)?;
    let v2_exists = path_entry_exists(&v2)?;
    if !v1_exists && !v2_exists {
        return Ok(None);
    }
    if v1_exists != v2_exists {
        return Err(InstallationError::Invalid(format!(
            "Cargo root {} has only one of {} and {}",
            root.display(),
            CARGO_V1_RECEIPT,
            CARGO_V2_RECEIPT
        )));
    }
    validate_root(root)?;
    let v1_bytes = read_controlled_file(&v1, root, MAX_RECEIPT_BYTES)?;
    let v2_bytes = read_controlled_file(&v2, root, MAX_RECEIPT_BYTES)?;
    let v1_owner = parse_v1_owner(&v1_bytes)?;
    let v2_record = parse_v2_owner(&v2_bytes)?;
    let (owner, options) = match (v1_owner, v2_record) {
        (None, None) => return Ok(None),
        (Some(v1), Some(v2)) if v1 == v2.owner => (v1, v2.options),
        (Some(_), Some(_)) => {
            return Err(InstallationError::Invalid(
                "Cargo receipts disagree on the hf2q package owner".to_owned(),
            ));
        }
        _ => {
            return Err(InstallationError::Invalid(
                "only one Cargo receipt claims the hf2q executable".to_owned(),
            ));
        }
    };
    let parsed = parse_package_id(&owner.package_id)?;
    if parsed.version.to_string() != expected_version {
        return Err(InstallationError::Invalid(format!(
            "Cargo receipt version {} does not match running hf2q {}",
            parsed.version, expected_version
        )));
    }
    Ok(Some((
        root.to_owned(),
        CargoRecord {
            version: parsed.version,
            source: parsed.source,
            options,
        },
    )))
}

pub(crate) fn reconcile_update(
    executable: &Path,
    expected_root: &Path,
    expected_source: &CargoSource,
    expected_options: &CargoInstallOptions,
) -> Result<String, InstallationError> {
    let output = Command::new(executable)
        .arg("--version")
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
        .map_err(|error| InstallationError::io("run updated hf2q --version", error))?;
    if !output.status.success()
        || output.stdout.is_empty()
        || output.stdout.len() > MAX_VERSION_OUTPUT_BYTES
    {
        return Err(InstallationError::Invalid(
            "updated Cargo executable did not report a bounded successful version".to_owned(),
        ));
    }
    let text = std::str::from_utf8(&output.stdout)
        .map_err(|_| InstallationError::Invalid("updated version is not UTF-8".to_owned()))?;
    let version = text
        .strip_prefix("hf2q ")
        .and_then(|text| text.strip_suffix('\n'))
        .ok_or_else(|| {
            InstallationError::Invalid("updated executable reported an invalid version".to_owned())
        })?;
    semver::Version::parse(version)
        .map_err(|_| InstallationError::Invalid("updated version is not SemVer".to_owned()))?;
    match detect_for_context(executable, version, Path::new(env!("CARGO_MANIFEST_DIR")))? {
        Installation::Cargo {
            root,
            source,
            options,
            ..
        } if root == expected_root
            && source.same_channel(expected_source)
            && options == *expected_options =>
        {
            Ok(version.to_owned())
        }
        _ => Err(InstallationError::Invalid(
            "updated executable is no longer owned by the same Cargo root, source selector, and install options"
                .to_owned(),
        )),
    }
}

pub(crate) fn reconcile_uninstall(root: &Path, executable: &Path) -> Result<(), InstallationError> {
    if path_entry_exists(executable)? {
        return Err(InstallationError::Invalid(format!(
            "Cargo reported success but {} still exists",
            executable.display()
        )));
    }
    let v1 = root.join(CARGO_V1_RECEIPT);
    let v2 = root.join(CARGO_V2_RECEIPT);
    let v1_exists = path_entry_exists(&v1)?;
    let v2_exists = path_entry_exists(&v2)?;
    if !v1_exists && !v2_exists {
        return Ok(());
    }
    if v1_exists != v2_exists {
        return Err(InstallationError::Invalid(
            "Cargo uninstall left only one tracking receipt".to_owned(),
        ));
    }
    validate_root(root)?;
    let v1_owner = parse_v1_owner(&read_controlled_file(&v1, root, MAX_RECEIPT_BYTES)?)?;
    let v2_owner = parse_v2_owner(&read_controlled_file(&v2, root, MAX_RECEIPT_BYTES)?)?;
    if v1_owner.is_some() || v2_owner.is_some() {
        return Err(InstallationError::Invalid(
            "Cargo reported success but its hf2q receipt remains".to_owned(),
        ));
    }
    Ok(())
}

fn validate_root(root: &Path) -> Result<(), InstallationError> {
    if root.as_os_str().as_bytes().len() > 1024
        || root
            .as_os_str()
            .as_bytes()
            .iter()
            .any(|byte| byte.is_ascii_control())
    {
        return Err(InstallationError::Invalid(
            "Cargo install root contains an unsafe path".to_owned(),
        ));
    }
    let canonical = fs::canonicalize(root)
        .map_err(|error| InstallationError::io("canonicalize Cargo install root", error))?;
    if canonical != root {
        return Err(InstallationError::Invalid(format!(
            "Cargo install root {} is not canonical",
            root.display()
        )));
    }
    let metadata = fs::metadata(root)
        .map_err(|error| InstallationError::io("inspect Cargo install root", error))?;
    if !metadata.is_dir()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.mode() & 0o022 != 0
    {
        return Err(InstallationError::Invalid(
            "Cargo install root must be current-user-owned and not group/world-writable".to_owned(),
        ));
    }
    Ok(())
}

fn read_controlled_file(
    path: &Path,
    root: &Path,
    maximum: u64,
) -> Result<Vec<u8>, InstallationError> {
    let root_metadata = fs::metadata(root)
        .map_err(|error| InstallationError::io("inspect evidence root", error))?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| InstallationError::io("open installation evidence", error))?;
    let before = file
        .metadata()
        .map_err(|error| InstallationError::io("inspect installation evidence", error))?;
    if !before.is_file()
        || before.uid() != rustix::process::geteuid().as_raw()
        || before.mode() & 0o022 != 0
        || before.nlink() != 1
        || before.dev() != root_metadata.dev()
        || before.len() == 0
        || before.len() > maximum
    {
        return Err(InstallationError::Invalid(format!(
            "{} is not a bounded current-user-controlled receipt",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    file.by_ref()
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| InstallationError::io("read installation evidence", error))?;
    if bytes.len() as u64 > maximum {
        return Err(InstallationError::Invalid(format!(
            "{} exceeds the receipt size bound",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| InstallationError::io("rewind installation evidence", error))?;
    let after = file
        .metadata()
        .map_err(|error| InstallationError::io("reinspect installation evidence", error))?;
    let named = fs::symlink_metadata(path)
        .map_err(|error| InstallationError::io("reopen installation evidence", error))?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.dev() != named.dev()
        || before.ino() != named.ino()
    {
        return Err(InstallationError::Invalid(format!(
            "{} changed while it was read",
            path.display()
        )));
    }
    Ok(bytes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CargoOwner {
    package_id: String,
    bins: BTreeSet<String>,
}

struct CargoV2Record {
    owner: CargoOwner,
    options: CargoInstallOptions,
}

fn parse_v1_owner(bytes: &[u8]) -> Result<Option<CargoOwner>, InstallationError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| InstallationError::Invalid(".crates.toml is not UTF-8".to_owned()))?;
    let value: toml::Value = toml::from_str(text)
        .map_err(|_| InstallationError::Invalid(".crates.toml is not valid TOML".to_owned()))?;
    let installs = value
        .get("v1")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| InstallationError::Invalid(".crates.toml has no v1 table".to_owned()))?;
    let mut owners = Vec::new();
    for (package_id, value) in installs {
        if !package_id.starts_with("hf2q ") {
            continue;
        }
        let bins = value.as_array().ok_or_else(|| {
            InstallationError::Invalid("hf2q v1 receipt bins are not an array".to_owned())
        })?;
        let bins = bins
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    InstallationError::Invalid(
                        "hf2q v1 receipt contains a non-string bin".to_owned(),
                    )
                })
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        owners.push(CargoOwner {
            package_id: package_id.to_owned(),
            bins,
        });
    }
    one_owner(owners, CARGO_V1_RECEIPT)
}

#[derive(Deserialize)]
struct CargoListingV2 {
    installs: BTreeMap<String, CargoInstallV2>,
}

#[derive(Deserialize)]
struct CargoInstallV2 {
    version_req: Option<String>,
    bins: Vec<String>,
    features: Vec<String>,
    all_features: bool,
    no_default_features: bool,
    profile: String,
    target: Option<String>,
}

fn parse_v2_owner(bytes: &[u8]) -> Result<Option<CargoV2Record>, InstallationError> {
    let listing: CargoListingV2 = serde_json::from_slice(bytes)
        .map_err(|_| InstallationError::Invalid(".crates2.json is not valid JSON".to_owned()))?;
    let mut installs = listing
        .installs
        .into_iter()
        .filter(|(package_id, _)| package_id.starts_with("hf2q "))
        .collect::<Vec<_>>();
    if installs.len() > 1 {
        return Err(InstallationError::Invalid(format!(
            "{CARGO_V2_RECEIPT} contains multiple hf2q package owners"
        )));
    }
    let Some((package_id, install)) = installs.pop() else {
        return Ok(None);
    };
    let bin_count = install.bins.len();
    let bins = install.bins.into_iter().collect::<BTreeSet<_>>();
    if bins.len() != bin_count || bins != BTreeSet::from(["hf2q".to_owned()]) {
        return Err(InstallationError::Invalid(format!(
            "{CARGO_V2_RECEIPT} does not bind exactly the hf2q executable"
        )));
    }
    let feature_count = install.features.len();
    let features = install.features.into_iter().collect::<BTreeSet<_>>();
    if features.len() != feature_count
        || features
            .iter()
            .any(|feature| !valid_argument(feature, 255) || feature.contains(','))
        || install
            .version_req
            .as_deref()
            .is_some_and(|value| !valid_argument(value, 255))
        || !valid_argument(&install.profile, 64)
        || install
            .target
            .as_deref()
            .is_some_and(|value| !valid_argument(value, 255))
    {
        return Err(InstallationError::Invalid(
            ".crates2.json contains invalid hf2q reinstall options".to_owned(),
        ));
    }
    Ok(Some(CargoV2Record {
        owner: CargoOwner { package_id, bins },
        options: CargoInstallOptions {
            version_req: install.version_req,
            features,
            all_features: install.all_features,
            no_default_features: install.no_default_features,
            profile: install.profile,
            target: install.target,
        },
    }))
}

fn valid_argument(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

fn one_owner(
    owners: Vec<CargoOwner>,
    receipt: &str,
) -> Result<Option<CargoOwner>, InstallationError> {
    if owners.len() > 1 {
        return Err(InstallationError::Invalid(format!(
            "{receipt} contains multiple hf2q package owners"
        )));
    }
    let Some(owner) = owners.into_iter().next() else {
        return Ok(None);
    };
    if owner.bins != BTreeSet::from(["hf2q".to_owned()]) {
        return Err(InstallationError::Invalid(format!(
            "{receipt} does not bind exactly the hf2q executable"
        )));
    }
    Ok(Some(owner))
}

struct ParsedPackageId {
    version: semver::Version,
    source: CargoSource,
}

fn parse_package_id(package_id: &str) -> Result<ParsedPackageId, InstallationError> {
    let body = package_id
        .strip_prefix("hf2q ")
        .ok_or_else(|| InstallationError::Invalid("Cargo package ID is not for hf2q".to_owned()))?;
    let source_start = body
        .rfind(" (")
        .ok_or_else(|| InstallationError::Invalid("Cargo package ID has no source".to_owned()))?;
    if !body.ends_with(')') {
        return Err(InstallationError::Invalid(
            "Cargo package ID has a malformed source".to_owned(),
        ));
    }
    let version_text = &body[..source_start];
    let source_text = &body[source_start + 2..body.len() - 1];
    let version = semver::Version::parse(version_text).map_err(|_| {
        InstallationError::Invalid("Cargo package ID version is not SemVer".to_owned())
    })?;
    if version.to_string() != version_text {
        return Err(InstallationError::Invalid(
            "Cargo package ID version is not canonical SemVer".to_owned(),
        ));
    }
    let source = if source_text == CRATES_IO_SOURCE {
        CargoSource::CratesIo
    } else if let Some(url_text) = source_text.strip_prefix("path+") {
        let url = url::Url::parse(url_text).map_err(|_| {
            InstallationError::Invalid("Cargo path source is not a valid URL".to_owned())
        })?;
        let path = url.to_file_path().map_err(|_| {
            InstallationError::Invalid("Cargo path source is not a local file URL".to_owned())
        })?;
        if path.as_os_str().as_bytes().len() > 4096
            || path
                .as_os_str()
                .as_bytes()
                .iter()
                .any(|byte| byte.is_ascii_control())
        {
            return Err(InstallationError::Invalid(
                "Cargo path source contains an unsafe path".to_owned(),
            ));
        }
        CargoSource::Path(path)
    } else if source_text.starts_with("git+") {
        parse_git_source(source_text)?
    } else if source_text.starts_with("registry+") {
        let index = source_text.strip_prefix("registry+").unwrap();
        let url = validate_source_url(index, "registry")?;
        if has_embedded_http_credentials(&url) {
            CargoSource::Other("credential-bearing registry source".to_owned())
        } else {
            CargoSource::OtherRegistry(source_text.to_owned())
        }
    } else {
        CargoSource::Other("unrecognized receipt source".to_owned())
    };
    Ok(ParsedPackageId { version, source })
}

fn parse_git_source(source_text: &str) -> Result<CargoSource, InstallationError> {
    let raw_url = source_text.strip_prefix("git+").ok_or_else(|| {
        InstallationError::Invalid("Cargo Git source has no git+ prefix".to_owned())
    })?;
    let mut url = validate_source_url(raw_url, "Git")?;
    if has_embedded_http_credentials(&url) {
        return Ok(CargoSource::Other(
            "credential-bearing Git source".to_owned(),
        ));
    }
    let resolved_revision = url.fragment().ok_or_else(|| {
        InstallationError::Invalid("Cargo Git source has no resolved revision".to_owned())
    })?;
    if !(7..=64).contains(&resolved_revision.len())
        || !resolved_revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(InstallationError::Invalid(
            "Cargo Git source has an invalid resolved revision".to_owned(),
        ));
    }
    let resolved_revision = resolved_revision.to_owned();
    let selectors = url
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    let selector = match selectors.as_slice() {
        [] => None,
        [(key, value)] if valid_argument(value, 255) => match key.as_str() {
            "branch" => Some(CargoGitSelector::Branch(value.to_owned())),
            "tag" => Some(CargoGitSelector::Tag(value.to_owned())),
            "rev" => Some(CargoGitSelector::Rev(value.to_owned())),
            _ => {
                return Err(InstallationError::Invalid(
                    "Cargo Git source has an unknown selector".to_owned(),
                ));
            }
        },
        _ => {
            return Err(InstallationError::Invalid(
                "Cargo Git source has multiple or invalid selectors".to_owned(),
            ));
        }
    };
    url.set_fragment(None);
    url.set_query(None);
    Ok(CargoSource::Git(CargoGitSource {
        repository: url.into(),
        selector,
        resolved_revision,
    }))
}

fn validate_source_url(value: &str, kind: &str) -> Result<url::Url, InstallationError> {
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(InstallationError::Invalid(format!(
            "Cargo {kind} source URL is invalid"
        )));
    }
    url::Url::parse(value)
        .map_err(|_| InstallationError::Invalid(format!("Cargo {kind} source URL is invalid")))
}

fn has_embedded_http_credentials(url: &url::Url) -> bool {
    url.password().is_some()
        || (matches!(url.scheme(), "http" | "https") && !url.username().is_empty())
}
