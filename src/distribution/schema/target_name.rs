use serde::{Serialize, Serializer};

use super::common::{ReleaseVersion, SchemaValueError, Sha256Digest, TargetTriple, UpdateChannel};

pub(crate) const MAX_TARGET_NAME_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalTargetKind {
    ChannelPointer,
    ReleaseManifest,
    ReleaseArchive,
}

/// A canonical consistent-snapshot object name derived from authenticated TUF
/// metadata, never from a pointer document or caller-supplied path.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ConsistentSnapshotTargetName {
    value: String,
}

impl ConsistentSnapshotTargetName {
    pub(crate) fn as_str(&self) -> &str {
        &self.value
    }

    /// GitHub Release assets are flat; only a route that has already selected
    /// the release-asset origin may consume this basename.
    pub(crate) fn basename(&self) -> &str {
        self.value
            .rsplit_once('/')
            .map_or(self.value.as_str(), |(_, basename)| basename)
    }
}

/// A canonical logical TUF target name.
///
/// Signed metadata and pointer documents always contain this unprefixed name.
/// A consistent-snapshot transport object name is derived separately from the
/// authenticated target digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalTargetName {
    value: String,
    kind: LogicalTargetKind,
    version: Option<ReleaseVersion>,
}

impl LogicalTargetName {
    pub(crate) fn channel_pointer(channel: UpdateChannel, target: TargetTriple) -> Self {
        Self {
            value: format!("channels/{}/{}.json", channel.as_str(), target.as_str()),
            kind: LogicalTargetKind::ChannelPointer,
            version: None,
        }
    }

    pub(crate) fn release_manifest(version: &ReleaseVersion, target: TargetTriple) -> Self {
        Self {
            value: format!(
                "releases/v{}/{}/release-manifest.json",
                version.as_str(),
                target.as_str()
            ),
            kind: LogicalTargetKind::ReleaseManifest,
            version: Some(version.clone()),
        }
    }

    pub(crate) fn release_archive(version: &ReleaseVersion, target: TargetTriple) -> Self {
        Self {
            value: format!(
                "releases/v{0}/{1}/hf2q-v{0}-{1}.zip",
                version.as_str(),
                target.as_str()
            ),
            kind: LogicalTargetKind::ReleaseArchive,
            version: Some(version.clone()),
        }
    }

    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        if value.is_empty()
            || value.len() > MAX_TARGET_NAME_BYTES
            || !value.is_ascii()
            || value.bytes().any(|byte| byte.is_ascii_control())
        {
            return Err(invalid_name(field));
        }

        let pointer =
            Self::channel_pointer(UpdateChannel::Stable, TargetTriple::Aarch64AppleDarwin);
        if value == pointer.value {
            return Ok(pointer);
        }

        let components: Vec<_> = value.split('/').collect();
        if components.len() != 4 || components[0] != "releases" {
            return Err(invalid_name(field));
        }
        let version_text = components[1]
            .strip_prefix('v')
            .ok_or_else(|| invalid_name(field))?;
        let version = ReleaseVersion::parse_stable(field, version_text.to_owned())?;
        let target = TargetTriple::parse(field, components[2].to_owned())?;
        let parsed = if components[3] == "release-manifest.json" {
            Self::release_manifest(&version, target)
        } else {
            Self::release_archive(&version, target)
        };
        if value != parsed.value {
            return Err(invalid_name(field));
        }
        Ok(parsed)
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.value
    }

    pub(crate) fn kind(&self) -> LogicalTargetKind {
        self.kind
    }

    pub(crate) fn version(&self) -> Option<&ReleaseVersion> {
        self.version.as_ref()
    }

    pub(crate) fn consistent_snapshot_name(
        &self,
        digest: &Sha256Digest,
    ) -> ConsistentSnapshotTargetName {
        let (parent, basename) = self
            .value
            .rsplit_once('/')
            .expect("canonical target names contain a parent");
        ConsistentSnapshotTargetName {
            value: format!("{parent}/{}.{basename}", digest.as_str()),
        }
    }
}

impl Serialize for LogicalTargetName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.value)
    }
}

fn invalid_name(field: &'static str) -> SchemaValueError {
    SchemaValueError::new(
        field,
        "must be a canonical hf2q stable pointer or versioned release target name",
    )
}
