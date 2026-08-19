use reqwest::Url;

use super::UpdateTransportError;
use crate::distribution::schema::{LogicalTargetKind, LogicalTargetName, ReleaseVersion};
use crate::distribution::update_auth::{
    AuthenticatedTargetDescriptor, MetadataRequestKind, MetadataRequestView,
};

const PAGES_METADATA_BASE: &str = "https://robertelee78.github.io/hf2q/updates/stable/metadata/";
const PAGES_TARGETS_BASE: &str = "https://robertelee78.github.io/hf2q/updates/stable/targets/";
const RELEASES_BASE: &str = "https://github.com/robertelee78/hf2q/releases/download/";
const RELEASE_CDN_HOST: &str = "release-assets.githubusercontent.com";
const MAX_LOCATION_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RequestClass {
    Pages,
    Release,
    ReleaseCdn,
}

pub(super) fn pages_metadata_url(
    spec: MetadataRequestView<'_>,
) -> Result<OriginLockedUrl, UpdateTransportError> {
    require_metadata_name(spec.kind(), spec.relative_name())?;
    let mut url =
        Url::parse(PAGES_METADATA_BASE).map_err(|_| UpdateTransportError::OriginPolicy)?;
    append_canonical_path(&mut url, spec.relative_name())?;
    require_initial_url(&url, "robertelee78.github.io")?;
    Ok(OriginLockedUrl {
        url,
        class: RequestClass::Pages,
    })
}

pub(super) struct OriginLockedUrl {
    url: Url,
    class: RequestClass,
}

impl std::fmt::Debug for OriginLockedUrl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OriginLockedUrl")
            .field("class", &self.class)
            .field("origin", &self.url.origin().ascii_serialization())
            .finish_non_exhaustive()
    }
}

impl OriginLockedUrl {
    pub(super) fn url(&self) -> &Url {
        &self.url
    }

    pub(super) fn class(&self) -> RequestClass {
        self.class
    }
}

pub(super) struct TargetFetchSpec<'a> {
    descriptor: &'a AuthenticatedTargetDescriptor,
    parsed_name: LogicalTargetName,
}

impl<'a> TargetFetchSpec<'a> {
    pub(super) fn from_descriptor(
        descriptor: &'a AuthenticatedTargetDescriptor,
    ) -> Result<Self, UpdateTransportError> {
        let parsed_name = LogicalTargetName::parse(
            "authenticated_target_name",
            descriptor.logical_name().to_owned(),
        )
        .map_err(|_| UpdateTransportError::OriginPolicy)?;
        Ok(Self {
            descriptor,
            parsed_name,
        })
    }

    pub(super) fn descriptor(&self) -> &AuthenticatedTargetDescriptor {
        self.descriptor
    }

    pub(super) fn kind(&self) -> LogicalTargetKind {
        self.parsed_name.kind()
    }

    pub(super) fn version(&self) -> Option<&ReleaseVersion> {
        self.parsed_name.version()
    }
}

pub(super) fn pages_pointer_url(
    spec: &TargetFetchSpec<'_>,
) -> Result<OriginLockedUrl, UpdateTransportError> {
    if spec.kind() != LogicalTargetKind::ChannelPointer {
        return Err(UpdateTransportError::OriginPolicy);
    }
    let mut url = Url::parse(PAGES_TARGETS_BASE).map_err(|_| UpdateTransportError::OriginPolicy)?;
    append_canonical_path(&mut url, spec.descriptor().physical_name().as_str())?;
    require_initial_url(&url, "robertelee78.github.io")?;
    Ok(OriginLockedUrl {
        url,
        class: RequestClass::Pages,
    })
}

pub(super) fn release_asset_url(
    release: &ReleaseVersion,
    spec: &TargetFetchSpec<'_>,
) -> Result<OriginLockedUrl, UpdateTransportError> {
    if !matches!(
        spec.kind(),
        LogicalTargetKind::ReleaseManifest | LogicalTargetKind::ReleaseArchive
    ) || spec.version() != Some(release)
    {
        return Err(UpdateTransportError::OriginPolicy);
    }
    let mut url = Url::parse(RELEASES_BASE).map_err(|_| UpdateTransportError::OriginPolicy)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| UpdateTransportError::OriginPolicy)?;
        segments.pop_if_empty();
        segments.push(&format!("v{}", release.as_str()));
        segments.push(spec.descriptor().physical_name().basename());
    }
    require_initial_url(&url, "github.com")?;
    Ok(OriginLockedUrl {
        url,
        class: RequestClass::Release,
    })
}

pub(super) fn release_redirect(location: &str) -> Result<OriginLockedUrl, UpdateTransportError> {
    if location.is_empty()
        || location.len() > MAX_LOCATION_BYTES
        || raw_authority_contains_userinfo(location)
    {
        return Err(UpdateTransportError::OriginPolicy);
    }
    let url = Url::parse(location).map_err(|_| UpdateTransportError::OriginPolicy)?;
    if url.scheme() != "https"
        || url.host_str() != Some(RELEASE_CDN_HOST)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || !matches!(url.port(), None | Some(443))
        || url.path().is_empty()
        || url.path() == "/"
    {
        return Err(UpdateTransportError::OriginPolicy);
    }
    if url
        .host_str()
        .is_some_and(|host| host.parse::<std::net::IpAddr>().is_ok())
    {
        return Err(UpdateTransportError::OriginPolicy);
    }
    Ok(OriginLockedUrl {
        url,
        class: RequestClass::ReleaseCdn,
    })
}

fn raw_authority_contains_userinfo(value: &str) -> bool {
    value
        .split_once("://")
        .and_then(|(_, remainder)| remainder.split(['/', '?', '#']).next())
        .is_some_and(|authority| authority.contains('@'))
}

fn append_canonical_path(url: &mut Url, path: &str) -> Result<(), UpdateTransportError> {
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| UpdateTransportError::OriginPolicy)?;
    segments.pop_if_empty();
    for component in path.split('/') {
        if component.is_empty() || component == "." || component == ".." {
            return Err(UpdateTransportError::OriginPolicy);
        }
        segments.push(component);
    }
    Ok(())
}

fn require_initial_url(url: &Url, host: &str) -> Result<(), UpdateTransportError> {
    if url.scheme() != "https"
        || url.host_str() != Some(host)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(UpdateTransportError::OriginPolicy);
    }
    Ok(())
}

fn require_metadata_name(
    kind: MetadataRequestKind,
    name: &str,
) -> Result<(), UpdateTransportError> {
    let valid = match kind {
        MetadataRequestKind::Root => versioned_metadata_name(name, "root"),
        MetadataRequestKind::Timestamp => name == "timestamp.json",
        MetadataRequestKind::Snapshot => {
            name == "snapshot.json" || versioned_metadata_name(name, "snapshot")
        }
        MetadataRequestKind::Targets => {
            name == "targets.json" || versioned_metadata_name(name, "targets")
        }
    };
    if valid {
        Ok(())
    } else {
        Err(UpdateTransportError::OriginPolicy)
    }
}

fn versioned_metadata_name(name: &str, role: &str) -> bool {
    let suffix = format!(".{role}.json");
    let Some(version) = name.strip_suffix(&suffix) else {
        return false;
    };
    version
        .parse::<u64>()
        .ok()
        .is_some_and(|parsed| parsed > 0 && parsed.to_string() == version)
}

#[cfg(test)]
pub(super) fn metadata_name_allowed_for_test(kind: MetadataRequestKind, name: &str) -> bool {
    require_metadata_name(kind, name).is_ok()
}
