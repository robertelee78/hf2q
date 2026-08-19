//! Canonical Hugging Face model references for ADR-045.
//!
//! This module is the sole parser for operator-supplied Hub repository IDs and
//! equivalent `huggingface.co` model, tree, blob, and resolve URLs. Parsing is
//! structural only. [`ResolvedHfModelReference`] is created only after the Hub
//! has returned an exact immutable commit for that parsed identity.

use percent_encoding::percent_decode_str;
use thiserror::Error;
use url::Url;

pub const MAX_HF_REFERENCE_BYTES: usize = 2 * 1024;
pub const MAX_HF_REPO_ID_BYTES: usize = 96;
pub const MAX_HF_REVISION_BYTES: usize = 255;
pub const MAX_HF_FILENAME_BYTES: usize = 1024;
pub const MAX_HF_FILENAME_COMPONENTS: usize = 64;

const CANONICAL_HF_ORIGIN: &str = "https://huggingface.co";

/// The only Hugging Face repository type accepted by the official-source
/// conversion path in ADR-045.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfRepositoryType {
    Model,
}

impl HfRepositoryType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
        }
    }
}

/// A bounded, normalized model reference that may still name a branch or tag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfModelReference {
    original: String,
    repo_id: String,
    canonical_url: String,
    requested_revision: Option<String>,
    filename: Option<String>,
}

/// The same model identity after the Hub has resolved it to an exact commit.
///
/// Fields are private so downstream download and receipt code cannot forge an
/// immutable identity from unrelated strings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedHfModelReference {
    original: String,
    repo_id: String,
    canonical_url: String,
    revision: String,
    filename: Option<String>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum HfReferenceError {
    #[error("Hugging Face reference is empty or exceeds {MAX_HF_REFERENCE_BYTES} bytes")]
    ReferenceSize,
    #[error("invalid Hugging Face model repository ID `{repo}`")]
    InvalidRepoId { repo: String },
    #[error("invalid Hugging Face revision `{revision}`")]
    InvalidRevision { revision: String },
    #[error("invalid Hugging Face filename `{filename}`")]
    InvalidFilename { filename: String },
    #[error("Hugging Face URL must use the exact origin {CANONICAL_HF_ORIGIN}")]
    InvalidOrigin,
    #[error("Hugging Face URL must not contain credentials, a port, query, or fragment")]
    UnexpectedUrlComponent,
    #[error("unsupported or ambiguous Hugging Face model URL route")]
    InvalidRoute,
    #[error("URL revision `{embedded}` does not match --revision `{explicit}`")]
    RevisionMismatch { embedded: String, explicit: String },
    #[error("Hub resolved `{resolved}` instead of an exact 40-hex commit")]
    InvalidResolvedCommit { resolved: String },
    #[error("Hugging Face URL contains invalid percent-encoded UTF-8")]
    InvalidPercentEncoding,
}

impl HfModelReference {
    /// Parse a repository ID or canonical Hugging Face URL and reconcile an
    /// optional CLI revision with any revision embedded in the URL.
    pub fn parse(input: &str, explicit_revision: Option<&str>) -> Result<Self, HfReferenceError> {
        validate_reference_input(input)?;
        let (repo_id, embedded_revision, filename) = if input.contains("://") {
            parse_model_url(input)?
        } else {
            validate_repo_id(input)?;
            (input.to_owned(), None, None)
        };

        let explicit_revision = explicit_revision.map(normalize_revision).transpose()?;
        let requested_revision = match (embedded_revision, explicit_revision) {
            (Some(embedded), Some(explicit)) if embedded != explicit => {
                return Err(HfReferenceError::RevisionMismatch { embedded, explicit });
            }
            (Some(embedded), _) => Some(embedded),
            (None, explicit) => explicit,
        };
        let canonical_url = format!("{CANONICAL_HF_ORIGIN}/{repo_id}");

        Ok(Self {
            original: input.to_owned(),
            repo_id,
            canonical_url,
            requested_revision,
            filename,
        })
    }

    pub fn original(&self) -> &str {
        &self.original
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub const fn repository_type(&self) -> HfRepositoryType {
        HfRepositoryType::Model
    }

    pub fn canonical_url(&self) -> &str {
        &self.canonical_url
    }

    pub fn requested_revision(&self) -> Option<&str> {
        self.requested_revision.as_deref()
    }

    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    /// Seal this parsed reference with the exact commit returned by Hub repo
    /// information. The commit is canonicalized to lowercase.
    pub fn resolve(
        self,
        resolved_commit: &str,
    ) -> Result<ResolvedHfModelReference, HfReferenceError> {
        let revision = normalize_resolved_commit(resolved_commit)?;
        Ok(ResolvedHfModelReference {
            original: self.original,
            repo_id: self.repo_id,
            canonical_url: self.canonical_url,
            revision,
            filename: self.filename,
        })
    }
}

impl ResolvedHfModelReference {
    pub fn original(&self) -> &str {
        &self.original
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub const fn repository_type(&self) -> HfRepositoryType {
        HfRepositoryType::Model
    }

    pub fn canonical_url(&self) -> &str {
        &self.canonical_url
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }
}

fn validate_reference_input(input: &str) -> Result<(), HfReferenceError> {
    if input.is_empty()
        || input.len() > MAX_HF_REFERENCE_BYTES
        || input.trim() != input
        || input
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
    {
        return Err(HfReferenceError::ReferenceSize);
    }
    Ok(())
}

fn parse_model_url(
    input: &str,
) -> Result<(String, Option<String>, Option<String>), HfReferenceError> {
    let url = Url::parse(input).map_err(|_| HfReferenceError::InvalidOrigin)?;
    if url.scheme() != "https" || url.host_str() != Some("huggingface.co") {
        return Err(HfReferenceError::InvalidOrigin);
    }

    let after_scheme = input
        .strip_prefix("https://")
        .ok_or(HfReferenceError::InvalidOrigin)?;
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    if authority != "huggingface.co"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(HfReferenceError::UnexpectedUrlComponent);
    }

    let raw_path = after_scheme
        .strip_prefix(authority)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or("");
    let mut raw_segments: Vec<&str> = raw_path.split('/').collect();
    if raw_segments.last() == Some(&"") {
        raw_segments.pop();
    }
    if raw_segments.is_empty()
        || raw_segments.len() > MAX_HF_FILENAME_COMPONENTS + 4
        || raw_segments.iter().any(|segment| segment.is_empty())
    {
        return Err(HfReferenceError::InvalidRoute);
    }

    let decoded = raw_segments
        .iter()
        .map(|segment| decode_segment(segment))
        .collect::<Result<Vec<_>, _>>()?;
    // A two-component repository may itself be named `tree`, `blob`, or
    // `resolve`. Treat those words as routes only when a route tail exists;
    // `owner/tree` must remain the canonical URL for that valid repository.
    let route_index = if decoded.len() > 2 && is_route(&decoded[1]) {
        Some(1)
    } else if decoded.len() > 2 && is_route(&decoded[2]) {
        Some(2)
    } else {
        None
    };

    match route_index {
        None if decoded.len() == 1 || decoded.len() == 2 => {
            let repo_id = decoded.join("/");
            validate_repo_id(&repo_id)?;
            Ok((repo_id, None, None))
        }
        Some(index) => {
            let repo_id = decoded[..index].join("/");
            validate_repo_id(&repo_id)?;
            let route = decoded[index].as_str();
            let tail = &decoded[index + 1..];
            match route {
                "tree" if tail.len() == 1 => {
                    let revision = normalize_revision(&tail[0])?;
                    Ok((repo_id, Some(revision), None))
                }
                "blob" | "resolve" if tail.len() >= 2 => {
                    let revision = normalize_revision(&tail[0])?;
                    if tail[1..]
                        .iter()
                        .any(|component| component.contains(['/', '\\']))
                    {
                        return Err(HfReferenceError::InvalidRoute);
                    }
                    let filename = tail[1..].join("/");
                    validate_filename(&filename)?;
                    Ok((repo_id, Some(revision), Some(filename)))
                }
                _ => Err(HfReferenceError::InvalidRoute),
            }
        }
        _ => Err(HfReferenceError::InvalidRoute),
    }
}

fn decode_segment(segment: &str) -> Result<String, HfReferenceError> {
    let bytes = segment.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return Err(HfReferenceError::InvalidPercentEncoding);
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    percent_decode_str(segment)
        .decode_utf8()
        .map(|decoded| decoded.into_owned())
        .map_err(|_| HfReferenceError::InvalidPercentEncoding)
}

fn is_route(segment: &str) -> bool {
    matches!(segment, "tree" | "blob" | "resolve")
}

fn validate_repo_id(repo: &str) -> Result<(), HfReferenceError> {
    let components: Vec<&str> = repo.split('/').collect();
    let valid = !repo.is_empty()
        && repo.len() <= MAX_HF_REPO_ID_BYTES
        && components.len() <= 2
        && !repo.contains("--")
        && !repo.contains("..")
        && !repo.ends_with(".git")
        && components.iter().all(|component| {
            !component.is_empty()
                && !component.starts_with(['-', '.'])
                && !component.ends_with(['-', '.'])
                && component
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        });
    if valid {
        Ok(())
    } else {
        Err(HfReferenceError::InvalidRepoId {
            repo: repo.to_owned(),
        })
    }
}

fn normalize_revision(revision: &str) -> Result<String, HfReferenceError> {
    let components: Vec<&str> = revision.split('/').collect();
    let valid = !revision.is_empty()
        && revision.len() <= MAX_HF_REVISION_BYTES
        && !revision.starts_with(['/', '.'])
        && !revision.ends_with(['/', '.'])
        && !revision.contains("//")
        && !revision.contains("..")
        && revision != "@"
        && !revision.contains("@{")
        && components.iter().all(|component| {
            !component.is_empty()
                && component != &"."
                && component != &".."
                && !component.starts_with('.')
                && !component.ends_with(".lock")
                && component
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        });
    if !valid {
        return Err(HfReferenceError::InvalidRevision {
            revision: revision.to_owned(),
        });
    }
    if is_exact_commit(revision) {
        Ok(revision.to_ascii_lowercase())
    } else {
        Ok(revision.to_owned())
    }
}

fn normalize_resolved_commit(commit: &str) -> Result<String, HfReferenceError> {
    if !is_exact_commit(commit) {
        return Err(HfReferenceError::InvalidResolvedCommit {
            resolved: commit.to_owned(),
        });
    }
    Ok(commit.to_ascii_lowercase())
}

fn is_exact_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_filename(filename: &str) -> Result<(), HfReferenceError> {
    let components: Vec<&str> = filename.split('/').collect();
    let valid = !filename.is_empty()
        && filename.len() <= MAX_HF_FILENAME_BYTES
        && components.len() <= MAX_HF_FILENAME_COMPONENTS
        && components.iter().all(|component| {
            !component.is_empty()
                && !matches!(*component, "." | "..")
                && component.trim() == *component
                && component.bytes().all(|byte| {
                    byte.is_ascii()
                        && !byte.is_ascii_control()
                        && !matches!(byte, b'/' | b'\\' | b'?' | b'#')
                })
        });
    if valid {
        Ok(())
    } else {
        Err(HfReferenceError::InvalidFilename {
            filename: filename.to_owned(),
        })
    }
}
