use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{MetadataJournalError, VerifiedMetadataCandidate};

pub(super) const GENERATION_KIND: &str = "hf2q.update-metadata-generation";
pub(super) const SELECTOR_KIND: &str = "hf2q.update-metadata-selector";
pub(super) const SCHEMA_VERSION: u32 = 1;
pub(super) const MAX_GENERATION_RECEIPT_BYTES: usize = 64 * 1024;
pub(super) const MAX_SELECTOR_BYTES: usize = 16 * 1024;
pub(super) const MAX_ROOT_CHAIN: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::distribution) struct MetadataGenerationReceiptV1 {
    kind: String,
    schema_version: u32,
    state_layout_schema: u32,
    package: String,
    sequence: u64,
    predecessor_generation_sha256: Option<String>,
    installation_id: String,
    state_root: String,
    repository_id: String,
    channel: String,
    verification_started_at: String,
    verification_completed_at: String,
    anchor_root: MetadataRoleDescriptorV1,
    root_chain: Vec<MetadataRoleDescriptorV1>,
    trusted_root: MetadataRoleDescriptorV1,
    timestamp: MetadataRoleDescriptorV1,
    snapshot: MetadataRoleDescriptorV1,
    targets: MetadataRoleDescriptorV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MetadataRoleDescriptorV1 {
    request_name: String,
    version: u64,
    length: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MetadataSelectorV1 {
    kind: String,
    schema_version: u32,
    sequence: u64,
    generation_sha256: String,
}

impl MetadataGenerationReceiptV1 {
    pub(super) fn new(
        sequence: u64,
        predecessor_generation_sha256: Option<String>,
        candidate: &VerifiedMetadataCandidate,
    ) -> Result<Self, MetadataJournalError> {
        let receipt = Self {
            kind: GENERATION_KIND.to_owned(),
            schema_version: SCHEMA_VERSION,
            state_layout_schema: 1,
            package: "hf2q".to_owned(),
            sequence,
            predecessor_generation_sha256,
            installation_id: candidate.installation_id().to_owned(),
            state_root: candidate.state_root().to_owned(),
            repository_id: candidate.repository_id().to_owned(),
            channel: candidate.channel().to_owned(),
            verification_started_at: candidate.verification_started_at().to_string(),
            verification_completed_at: candidate.verification_completed_at().to_string(),
            anchor_root: descriptor(candidate.anchor_root()),
            root_chain: candidate.root_chain().iter().map(descriptor).collect(),
            trusted_root: descriptor(candidate.trusted_root()),
            timestamp: descriptor(candidate.timestamp()),
            snapshot: descriptor(candidate.snapshot()),
            targets: descriptor(candidate.targets()),
        };
        // Construction and hostile parsing deliberately share one invariant
        // path so an in-memory value cannot be committed and then rejected.
        Self::parse(&receipt.to_bytes()?)
    }

    pub(in crate::distribution) fn parse(bytes: &[u8]) -> Result<Self, MetadataJournalError> {
        require_bound(bytes, MAX_GENERATION_RECEIPT_BYTES, "generation receipt")?;
        let receipt: Self = serde_json::from_slice(bytes)
            .map_err(|_| MetadataJournalError::Invalid("generation receipt JSON is invalid"))?;
        receipt.validate_invariants()?;
        if receipt.to_bytes()? != bytes {
            return Err(MetadataJournalError::Invalid(
                "generation receipt is not canonical v1 JSON",
            ));
        }
        Ok(receipt)
    }

    pub(super) fn to_bytes(&self) -> Result<Vec<u8>, MetadataJournalError> {
        let mut bytes = serde_json::to_vec(self).map_err(|_| {
            MetadataJournalError::Invalid("generation receipt cannot be serialized")
        })?;
        bytes.push(b'\n');
        require_bound(&bytes, MAX_GENERATION_RECEIPT_BYTES, "generation receipt")?;
        Ok(bytes)
    }

    pub(in crate::distribution) fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(in crate::distribution) fn verification_started_at(
        &self,
    ) -> Result<jiff::Timestamp, MetadataJournalError> {
        parse_canonical_time(&self.verification_started_at)
    }

    pub(in crate::distribution) fn verification_completed_at(
        &self,
    ) -> Result<jiff::Timestamp, MetadataJournalError> {
        parse_canonical_time(&self.verification_completed_at)
    }

    pub(in crate::distribution) fn validate_authenticated_role(
        &self,
        stored_name: &str,
        request_name: &str,
        version: u64,
        bytes: &[u8],
    ) -> Result<(), MetadataJournalError> {
        let descriptor = match stored_name {
            "anchor-root.json" => &self.anchor_root,
            "trusted-root.json" => &self.trusted_root,
            "timestamp.json" => &self.timestamp,
            "snapshot.json" => &self.snapshot,
            "targets.json" => &self.targets,
            _ => {
                return Err(MetadataJournalError::Invalid(
                    "unknown authenticated metadata role",
                ))
            }
        };
        if descriptor.request_name != request_name || descriptor.version != version {
            return Err(MetadataJournalError::Invalid(
                "authenticated metadata identity differs from its receipt",
            ));
        }
        validate_descriptor_bytes(descriptor, bytes)
    }

    pub(in crate::distribution) fn validate_authenticated_root(
        &self,
        index: usize,
        request_name: &str,
        version: u64,
        bytes: &[u8],
    ) -> Result<(), MetadataJournalError> {
        let descriptor = self
            .root_chain
            .get(index)
            .ok_or(MetadataJournalError::Invalid(
                "authenticated root is absent from its receipt",
            ))?;
        if descriptor.request_name != request_name || descriptor.version != version {
            return Err(MetadataJournalError::Invalid(
                "authenticated root identity differs from its receipt",
            ));
        }
        validate_descriptor_bytes(descriptor, bytes)
    }

    pub(super) fn digest(&self) -> Result<String, MetadataJournalError> {
        Ok(hex::encode(Sha256::digest(self.to_bytes()?)))
    }

    pub(super) fn expected_root_names(&self) -> Vec<String> {
        self.root_chain
            .iter()
            .map(|root| format!("{:020}.root.json", root.version))
            .collect()
    }

    pub(super) fn predecessor_digest(&self) -> Option<&str> {
        self.predecessor_generation_sha256.as_deref()
    }

    pub(in crate::distribution) fn validate_state_identity(
        &self,
        installation_id: &str,
        state_root: &str,
    ) -> Result<(), MetadataJournalError> {
        if self.installation_id != installation_id || self.state_root != state_root {
            return Err(MetadataJournalError::Invalid(
                "metadata generation belongs to a different installation state root",
            ));
        }
        Ok(())
    }

    pub(in crate::distribution) fn matches_candidate(
        &self,
        candidate: &VerifiedMetadataCandidate,
    ) -> bool {
        self.installation_id == candidate.installation_id()
            && self.state_root == candidate.state_root()
            && self.repository_id == candidate.repository_id()
            && self.channel == candidate.channel()
            && self.verification_started_at == candidate.verification_started_at().to_string()
            && self.verification_completed_at == candidate.verification_completed_at().to_string()
            && descriptor_matches(&self.anchor_root, candidate.anchor_root())
            && self.root_chain.len() == candidate.root_chain().len()
            && self
                .root_chain
                .iter()
                .zip(candidate.root_chain())
                .all(|(stored, actual)| descriptor_matches(stored, actual))
            && descriptor_matches(&self.trusted_root, candidate.trusted_root())
            && descriptor_matches(&self.timestamp, candidate.timestamp())
            && descriptor_matches(&self.snapshot, candidate.snapshot())
            && descriptor_matches(&self.targets, candidate.targets())
    }

    pub(super) fn validate_successor(
        &self,
        prior: &Self,
        prior_digest: &str,
    ) -> Result<(), MetadataJournalError> {
        if self.sequence
            != prior
                .sequence
                .checked_add(1)
                .ok_or(MetadataJournalError::Invalid(
                    "metadata generation sequence overflowed",
                ))?
            || self.predecessor_digest() != Some(prior_digest)
            || self.installation_id != prior.installation_id
            || self.state_root != prior.state_root
            || self.repository_id != prior.repository_id
            || self.channel != prior.channel
            || self.anchor_root != prior.anchor_root
        {
            return Err(MetadataJournalError::Invalid(
                "metadata generation predecessor is not exact",
            ));
        }
        if self.root_chain.len() < prior.root_chain.len()
            || self.root_chain[..prior.root_chain.len()] != prior.root_chain
        {
            return Err(MetadataJournalError::Invalid(
                "metadata root history changed below its trusted floor",
            ));
        }

        let prior_time = parse_canonical_time(&prior.verification_completed_at)?;
        let started = parse_canonical_time(&self.verification_started_at)?;
        let completed = parse_canonical_time(&self.verification_completed_at)?;
        if started < prior_time || completed < started {
            return Err(MetadataJournalError::Invalid(
                "metadata verification clock floor moved backward",
            ));
        }
        for (old, new) in [
            (&prior.trusted_root, &self.trusted_root),
            (&prior.timestamp, &self.timestamp),
            (&prior.snapshot, &self.snapshot),
            (&prior.targets, &self.targets),
        ] {
            if new.version < old.version || (new.version == old.version && new.sha256 != old.sha256)
            {
                return Err(MetadataJournalError::Invalid(
                    "metadata role floor moved backward or equivocated",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn validate_role_bytes(
        &self,
        stored_name: &str,
        bytes: &[u8],
    ) -> Result<(), MetadataJournalError> {
        let descriptor = match stored_name {
            "anchor-root.json" => &self.anchor_root,
            "trusted-root.json" => &self.trusted_root,
            "timestamp.json" => &self.timestamp,
            "snapshot.json" => &self.snapshot,
            "targets.json" => &self.targets,
            _ => {
                return Err(MetadataJournalError::Invalid(
                    "unknown stored metadata role",
                ))
            }
        };
        validate_descriptor_bytes(descriptor, bytes)
    }

    pub(super) fn validate_root_bytes(
        &self,
        stored_name: &str,
        bytes: &[u8],
    ) -> Result<(), MetadataJournalError> {
        let descriptor = self
            .root_chain
            .iter()
            .find(|root| format!("{:020}.root.json", root.version) == stored_name)
            .ok_or(MetadataJournalError::Invalid(
                "unexpected stored root-chain role",
            ))?;
        validate_descriptor_bytes(descriptor, bytes)
    }

    pub(super) fn role_limit(&self, stored_name: &str) -> Result<usize, MetadataJournalError> {
        let length = match stored_name {
            "anchor-root.json" => self.anchor_root.length,
            "trusted-root.json" => self.trusted_root.length,
            "timestamp.json" => self.timestamp.length,
            "snapshot.json" => self.snapshot.length,
            "targets.json" => self.targets.length,
            _ => {
                return Err(MetadataJournalError::Invalid(
                    "unknown stored metadata role",
                ))
            }
        };
        bounded_length(length)
    }

    pub(super) fn root_limit(&self, stored_name: &str) -> Result<usize, MetadataJournalError> {
        let length = self
            .root_chain
            .iter()
            .find(|root| format!("{:020}.root.json", root.version) == stored_name)
            .ok_or(MetadataJournalError::Invalid(
                "unexpected stored root-chain role",
            ))?
            .length;
        bounded_length(length)
    }

    fn validate_invariants(&self) -> Result<(), MetadataJournalError> {
        if self.kind != GENERATION_KIND
            || self.schema_version != SCHEMA_VERSION
            || self.state_layout_schema != 1
            || self.package != "hf2q"
            || self.sequence == 0
        {
            return Err(MetadataJournalError::Invalid(
                "generation receipt envelope is unsupported",
            ));
        }
        if (self.sequence == 1) != self.predecessor_generation_sha256.is_none()
            || self
                .predecessor_generation_sha256
                .as_deref()
                .is_some_and(|value| !is_sha256(value))
        {
            return Err(MetadataJournalError::Invalid(
                "generation predecessor digest is invalid",
            ));
        }
        validate_installation_identity(&self.installation_id, &self.state_root)?;
        if self.repository_id != "hf2q" || self.channel != "stable" {
            return Err(MetadataJournalError::Invalid(
                "generation repository or channel is unsupported",
            ));
        }
        let started = parse_canonical_time(&self.verification_started_at)?;
        let completed = parse_canonical_time(&self.verification_completed_at)?;
        if completed < started {
            return Err(MetadataJournalError::Invalid(
                "verification completion precedes its start",
            ));
        }
        if self.root_chain.len() > MAX_ROOT_CHAIN {
            return Err(MetadataJournalError::Invalid(
                "root history exceeds the v1 lifetime bound",
            ));
        }
        validate_descriptor(&self.anchor_root, RoleKind::Root)?;
        validate_descriptor(&self.trusted_root, RoleKind::Root)?;
        validate_descriptor(&self.timestamp, RoleKind::Timestamp)?;
        validate_descriptor(&self.snapshot, RoleKind::Snapshot)?;
        validate_descriptor(&self.targets, RoleKind::Targets)?;

        let expected_chain_len = self
            .trusted_root
            .version
            .checked_sub(self.anchor_root.version)
            .ok_or(MetadataJournalError::Invalid(
                "trusted root predates its embedded anchor",
            ))? as usize;
        if expected_chain_len != self.root_chain.len() {
            return Err(MetadataJournalError::Invalid(
                "root history is not gapless from the embedded anchor",
            ));
        }
        for (offset, root) in self.root_chain.iter().enumerate() {
            validate_descriptor(root, RoleKind::Root)?;
            let expected_version = self
                .anchor_root
                .version
                .checked_add(offset as u64 + 1)
                .ok_or(MetadataJournalError::Invalid(
                    "root history version overflowed",
                ))?;
            if root.version != expected_version {
                return Err(MetadataJournalError::Invalid(
                    "root history is not sequential",
                ));
            }
        }
        if let Some(last) = self.root_chain.last() {
            if last != &self.trusted_root {
                return Err(MetadataJournalError::Invalid(
                    "root history does not end at the trusted root",
                ));
            }
        } else if self.anchor_root != self.trusted_root {
            return Err(MetadataJournalError::Invalid(
                "unchanged trusted root differs from its anchor",
            ));
        }
        Ok(())
    }
}

impl MetadataSelectorV1 {
    pub(super) fn new(
        sequence: u64,
        generation_sha256: String,
    ) -> Result<Self, MetadataJournalError> {
        let selector = Self {
            kind: SELECTOR_KIND.to_owned(),
            schema_version: SCHEMA_VERSION,
            sequence,
            generation_sha256,
        };
        Self::parse(&selector.to_bytes()?)
    }

    pub(super) fn parse(bytes: &[u8]) -> Result<Self, MetadataJournalError> {
        require_bound(bytes, MAX_SELECTOR_BYTES, "metadata selector")?;
        let selector: Self = serde_json::from_slice(bytes)
            .map_err(|_| MetadataJournalError::Invalid("metadata selector JSON is invalid"))?;
        if selector.kind != SELECTOR_KIND
            || selector.schema_version != SCHEMA_VERSION
            || selector.sequence == 0
            || !is_sha256(&selector.generation_sha256)
            || selector.to_bytes()? != bytes
        {
            return Err(MetadataJournalError::Invalid(
                "metadata selector is not canonical v1",
            ));
        }
        Ok(selector)
    }

    pub(super) fn to_bytes(&self) -> Result<Vec<u8>, MetadataJournalError> {
        let mut bytes = serde_json::to_vec(self)
            .map_err(|_| MetadataJournalError::Invalid("metadata selector cannot serialize"))?;
        bytes.push(b'\n');
        require_bound(&bytes, MAX_SELECTOR_BYTES, "metadata selector")?;
        Ok(bytes)
    }

    pub(super) fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) fn generation_sha256(&self) -> &str {
        &self.generation_sha256
    }
}

#[derive(Clone, Copy)]
enum RoleKind {
    Root,
    Timestamp,
    Snapshot,
    Targets,
}

fn descriptor(role: &super::ExactMetadataRole) -> MetadataRoleDescriptorV1 {
    MetadataRoleDescriptorV1 {
        request_name: role.request_name().to_owned(),
        version: role.version(),
        length: role.bytes().len() as u64,
        sha256: hex::encode(Sha256::digest(role.bytes())),
    }
}

fn descriptor_matches(
    descriptor: &MetadataRoleDescriptorV1,
    role: &super::ExactMetadataRole,
) -> bool {
    descriptor == &self::descriptor(role)
}

fn validate_descriptor(
    descriptor: &MetadataRoleDescriptorV1,
    role: RoleKind,
) -> Result<(), MetadataJournalError> {
    if descriptor.version == 0 || !is_sha256(&descriptor.sha256) {
        return Err(MetadataJournalError::Invalid(
            "metadata role descriptor is invalid",
        ));
    }
    let maximum = match role {
        RoleKind::Targets => 4 * 1024 * 1024,
        _ => 1024 * 1024,
    };
    if descriptor.length == 0 || descriptor.length > maximum {
        return Err(MetadataJournalError::Invalid(
            "metadata role length is outside its bound",
        ));
    }
    let expected = match role {
        RoleKind::Root => format!("{}.root.json", descriptor.version),
        RoleKind::Timestamp => "timestamp.json".to_owned(),
        RoleKind::Snapshot => {
            if descriptor.request_name == "snapshot.json" {
                "snapshot.json".to_owned()
            } else {
                format!("{}.snapshot.json", descriptor.version)
            }
        }
        RoleKind::Targets => {
            if descriptor.request_name == "targets.json" {
                "targets.json".to_owned()
            } else {
                format!("{}.targets.json", descriptor.version)
            }
        }
    };
    if descriptor.request_name != expected {
        return Err(MetadataJournalError::Invalid(
            "metadata request name and role version disagree",
        ));
    }
    Ok(())
}

fn validate_descriptor_bytes(
    descriptor: &MetadataRoleDescriptorV1,
    bytes: &[u8],
) -> Result<(), MetadataJournalError> {
    if bytes.len() as u64 != descriptor.length
        || hex::encode(Sha256::digest(bytes)) != descriptor.sha256
    {
        return Err(MetadataJournalError::Invalid(
            "stored metadata bytes do not match their receipt descriptor",
        ));
    }
    Ok(())
}

fn validate_installation_identity(
    installation_id: &str,
    state_root: &str,
) -> Result<(), MetadataJournalError> {
    let uuid = uuid::Uuid::parse_str(installation_id).map_err(|_| {
        MetadataJournalError::Invalid("metadata generation installation ID is invalid")
    })?;
    if uuid.hyphenated().to_string() != installation_id
        || uuid.get_version() != Some(uuid::Version::Random)
        || uuid.get_variant() != uuid::Variant::RFC4122
    {
        return Err(MetadataJournalError::Invalid(
            "metadata generation installation ID is invalid",
        ));
    }
    let parsed =
        super::super::schema::AbsoluteInstallPath::parse("state_root", state_root.to_owned())
            .map_err(|_| {
                MetadataJournalError::Invalid("metadata generation state root is invalid")
            })?;
    if parsed.as_str() != state_root {
        return Err(MetadataJournalError::Invalid(
            "metadata generation state root is not canonical",
        ));
    }
    Ok(())
}

fn parse_canonical_time(value: &str) -> Result<jiff::Timestamp, MetadataJournalError> {
    let timestamp = value
        .parse::<jiff::Timestamp>()
        .map_err(|_| MetadataJournalError::Invalid("metadata timestamp is invalid"))?;
    if timestamp.to_string() != value {
        return Err(MetadataJournalError::Invalid(
            "metadata timestamp is not canonical RFC 3339",
        ));
    }
    Ok(timestamp)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn require_bound(
    bytes: &[u8],
    maximum: usize,
    label: &'static str,
) -> Result<(), MetadataJournalError> {
    if bytes.len() > maximum {
        return Err(MetadataJournalError::Invalid(match label {
            "generation receipt" => "generation receipt exceeds its input bound",
            _ => "metadata selector exceeds its input bound",
        }));
    }
    Ok(())
}

fn bounded_length(length: u64) -> Result<usize, MetadataJournalError> {
    usize::try_from(length)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or(MetadataJournalError::Invalid(
            "stored metadata length exceeds this platform",
        ))
}
