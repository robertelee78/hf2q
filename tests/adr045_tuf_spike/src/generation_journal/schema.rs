use serde::{Deserialize, Serialize};

use super::{CandidateGeneration, JournalError};
use crate::model::{
    sha256, CapturedRole, MAX_ROOT_BYTES, MAX_SNAPSHOT_BYTES, MAX_TARGETS_BYTES,
    MAX_TIMESTAMP_BYTES,
};

pub(super) const MAX_RETAINED_GENERATIONS: u64 = 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationReceiptV0 {
    schema_version: u64,
    pub(super) sequence: u64,
    predecessor_generation_sha256: Option<String>,
    repository: String,
    channel: String,
    update_start: String,
    prior_root: RoleDescriptorV0,
    root_chain: Vec<RoleDescriptorV0>,
    root: RoleDescriptorV0,
    timestamp: RoleDescriptorV0,
    snapshot: RoleDescriptorV0,
    targets: RoleDescriptorV0,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RoleDescriptorV0 {
    request_name: String,
    version: u64,
    length: u64,
    sha256: String,
}

impl GenerationReceiptV0 {
    pub(super) fn new(
        sequence: u64,
        predecessor_generation_sha256: Option<String>,
        candidate: &CandidateGeneration,
    ) -> Result<Self, JournalError> {
        let value = Self {
            schema_version: 0,
            sequence,
            predecessor_generation_sha256,
            repository: candidate.repository.clone(),
            channel: candidate.channel.clone(),
            update_start: candidate.update_start.to_string(),
            prior_root: descriptor(&candidate.prior_root),
            root_chain: candidate.root_chain.iter().map(descriptor).collect(),
            root: descriptor(&candidate.root),
            timestamp: descriptor(&candidate.timestamp),
            snapshot: descriptor(&candidate.snapshot),
            targets: descriptor(&candidate.targets),
        };
        Self::parse(&value.to_bytes()?)
    }

    pub(super) fn to_bytes(&self) -> Result<Vec<u8>, JournalError> {
        let mut bytes = serde_json::to_vec(self)
            .map_err(|_| JournalError::Invalid("generation receipt cannot serialize"))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub(super) fn parse(bytes: &[u8]) -> Result<Self, JournalError> {
        if bytes.len() > 64 * 1024 {
            return Err(JournalError::Invalid(
                "generation receipt exceeds its bound",
            ));
        }
        crate::strict_json::validate(bytes, 64 * 1024)
            .map_err(|_| JournalError::Invalid("generation receipt JSON is invalid"))?;
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|_| JournalError::Invalid("generation receipt schema is invalid"))?;
        if !value.invariants_hold() || value.to_bytes()? != bytes {
            return Err(JournalError::Invalid(
                "generation receipt is not canonical v0",
            ));
        }
        Ok(value)
    }

    pub(super) fn validate_candidate(
        &self,
        candidate: &CandidateGeneration,
    ) -> Result<(), JournalError> {
        if !self.matches_candidate(candidate) {
            return Err(JournalError::Invalid(
                "generation receipt does not bind candidate bytes",
            ));
        }
        Ok(())
    }

    pub(super) fn matches_candidate(&self, candidate: &CandidateGeneration) -> bool {
        self.repository == candidate.repository
            && self.channel == candidate.channel
            && self.update_start == candidate.update_start.to_string()
            && descriptor_matches(&self.prior_root, &candidate.prior_root)
            && self.root_chain.len() == candidate.root_chain.len()
            && self
                .root_chain
                .iter()
                .zip(&candidate.root_chain)
                .all(|(descriptor, role)| descriptor_matches(descriptor, role))
            && descriptor_matches(&self.root, &candidate.root)
            && descriptor_matches(&self.timestamp, &candidate.timestamp)
            && descriptor_matches(&self.snapshot, &candidate.snapshot)
            && descriptor_matches(&self.targets, &candidate.targets)
    }

    pub(super) fn validate_successor(
        &self,
        prior: &Self,
        prior_receipt_sha256: &str,
    ) -> Result<(), JournalError> {
        if self.sequence
            != prior
                .sequence
                .checked_add(1)
                .ok_or(JournalError::Invalid("generation sequence overflowed"))?
            || self.predecessor_generation_sha256.as_deref() != Some(prior_receipt_sha256)
            || self.repository != prior.repository
            || self.channel != prior.channel
            || self.prior_root != prior.root
        {
            return Err(JournalError::Invalid("generation predecessor is not exact"));
        }
        let prior_time = prior
            .update_start
            .parse::<jiff::Timestamp>()
            .map_err(|_| JournalError::Invalid("prior clock floor is invalid"))?;
        let current_time = self
            .update_start
            .parse::<jiff::Timestamp>()
            .map_err(|_| JournalError::Invalid("current clock floor is invalid"))?;
        if current_time < prior_time {
            return Err(JournalError::Invalid("clock floor moved backward"));
        }
        for (old, new) in [
            (&prior.root, &self.root),
            (&prior.timestamp, &self.timestamp),
            (&prior.snapshot, &self.snapshot),
            (&prior.targets, &self.targets),
        ] {
            if new.version < old.version || (new.version == old.version && new.sha256 != old.sha256)
            {
                return Err(JournalError::Invalid(
                    "role floor moved backward or equivocated",
                ));
            }
        }
        if self.root.version == self.prior_root.version {
            if !self.root_chain.is_empty() || self.root.sha256 != prior.root.sha256 {
                return Err(JournalError::Invalid(
                    "unchanged root has a conflicting chain",
                ));
            }
        } else {
            let expected_len = self
                .root
                .version
                .checked_sub(self.prior_root.version)
                .ok_or(JournalError::Invalid("root version moved backward"))?;
            if expected_len as usize != self.root_chain.len() {
                return Err(JournalError::Invalid("root chain is not gapless"));
            }
            for (offset, root) in self.root_chain.iter().enumerate() {
                let expected = self
                    .prior_root
                    .version
                    .checked_add(offset as u64 + 1)
                    .ok_or(JournalError::Invalid("root version overflowed"))?;
                if root.version != expected {
                    return Err(JournalError::Invalid("root chain is not sequential"));
                }
            }
            if self
                .root_chain
                .last()
                .map(|root| (&root.sha256, root.version))
                != Some((&self.root.sha256, self.root.version))
            {
                return Err(JournalError::Invalid(
                    "root chain does not end at trusted root",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) fn expected_root_chain_names(&self) -> Vec<String> {
        self.root_chain
            .iter()
            .map(|root| format!("{:020}.root.json", root.version))
            .collect()
    }

    pub(super) fn validate_stored_role(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> Result<(), JournalError> {
        let descriptor = match name {
            "trusted-root-before.json" => &self.prior_root,
            "trusted-root.json" => &self.root,
            "timestamp.json" => &self.timestamp,
            "snapshot.json" => &self.snapshot,
            "targets.json" => &self.targets,
            _ => return Err(JournalError::Invalid("unknown stored metadata role")),
        };
        validate_descriptor_bytes(descriptor, bytes)
    }

    pub(super) fn stored_role_limit(&self, name: &str) -> Result<usize, JournalError> {
        let length = match name {
            "trusted-root-before.json" => self.prior_root.length,
            "trusted-root.json" => self.root.length,
            "timestamp.json" => self.timestamp.length,
            "snapshot.json" => self.snapshot.length,
            "targets.json" => self.targets.length,
            _ => return Err(JournalError::Invalid("unknown stored metadata role")),
        };
        usize::try_from(length)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(JournalError::Invalid(
                "stored role length exceeds the platform",
            ))
    }

    pub(super) fn validate_stored_root(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> Result<(), JournalError> {
        let descriptor = self
            .root_chain
            .iter()
            .find(|root| format!("{:020}.root.json", root.version) == name)
            .ok_or(JournalError::Invalid("unexpected root chain entry"))?;
        validate_descriptor_bytes(descriptor, bytes)
    }

    pub(super) fn stored_root_limit(&self, name: &str) -> Result<usize, JournalError> {
        let descriptor = self
            .root_chain
            .iter()
            .find(|root| format!("{:020}.root.json", root.version) == name)
            .ok_or(JournalError::Invalid("unexpected root chain entry"))?;
        usize::try_from(descriptor.length)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(JournalError::Invalid(
                "root chain length exceeds the platform",
            ))
    }

    fn invariants_hold(&self) -> bool {
        if self.schema_version != 0
            || self.sequence == 0
            || self.sequence > MAX_RETAINED_GENERATIONS
            || !valid_repository(&self.repository)
            || !valid_channel(&self.channel)
            || self.root_chain.len() > 32
            || !descriptor_is_bounded(&self.prior_root, MAX_ROOT_BYTES)
            || !descriptor_is_bounded(&self.root, MAX_ROOT_BYTES)
            || !descriptor_is_bounded(&self.timestamp, MAX_TIMESTAMP_BYTES)
            || !descriptor_is_bounded(&self.snapshot, MAX_SNAPSHOT_BYTES)
            || !descriptor_is_bounded(&self.targets, MAX_TARGETS_BYTES)
            || !root_request_matches(&self.prior_root)
            || !root_request_matches(&self.root)
            || self.timestamp.request_name != "timestamp.json"
            || !lower_request_matches(&self.snapshot, "snapshot.json")
            || !lower_request_matches(&self.targets, "targets.json")
            || self
                .root_chain
                .iter()
                .any(|root| !descriptor_is_bounded(root, MAX_ROOT_BYTES))
            || !canonical_timestamp(&self.update_start)
            || !predecessor_shape(self.sequence, self.predecessor_generation_sha256.as_deref())
        {
            return false;
        }
        if self.root_chain.is_empty() {
            return self.prior_root == self.root;
        }
        for (offset, root) in self.root_chain.iter().enumerate() {
            let Some(expected) = self.prior_root.version.checked_add(offset as u64 + 1) else {
                return false;
            };
            if root.version != expected || !root_request_matches(root) {
                return false;
            }
        }
        self.root_chain.last() == Some(&self.root)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SelectorV0 {
    schema_version: u64,
    pub(super) sequence: u64,
    pub(super) generation_sha256: String,
}

impl SelectorV0 {
    pub(super) fn new(sequence: u64, generation_sha256: String) -> Self {
        Self {
            schema_version: 0,
            sequence,
            generation_sha256,
        }
    }

    pub(super) fn to_bytes(&self) -> Result<Vec<u8>, JournalError> {
        let mut bytes = serde_json::to_vec(self)
            .map_err(|_| JournalError::Invalid("selector cannot serialize"))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub(super) fn parse(bytes: &[u8]) -> Result<Self, JournalError> {
        crate::strict_json::validate(bytes, 16 * 1024)
            .map_err(|_| JournalError::Invalid("selector JSON is invalid"))?;
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|_| JournalError::Invalid("selector schema is invalid"))?;
        if value.schema_version != 0
            || value.sequence == 0
            || value.sequence > MAX_RETAINED_GENERATIONS
            || value.generation_sha256.len() != 64
            || hex::decode(&value.generation_sha256).map_or(true, |bytes| bytes.len() != 32)
            || value.generation_sha256 != value.generation_sha256.to_ascii_lowercase()
            || value.to_bytes()? != bytes
        {
            return Err(JournalError::Invalid("selector is not canonical v0"));
        }
        Ok(value)
    }
}

fn descriptor(role: &CapturedRole) -> RoleDescriptorV0 {
    RoleDescriptorV0 {
        request_name: role.request_name.clone(),
        version: role.version,
        length: role.raw.len() as u64,
        sha256: hex::encode(sha256(&role.raw)),
    }
}

fn descriptor_matches(descriptor: &RoleDescriptorV0, role: &CapturedRole) -> bool {
    descriptor.request_name == role.request_name
        && descriptor.version == role.version
        && descriptor.length == role.raw.len() as u64
        && descriptor.sha256 == hex::encode(sha256(&role.raw))
}

fn descriptor_is_bounded(descriptor: &RoleDescriptorV0, max_bytes: usize) -> bool {
    !descriptor.request_name.is_empty()
        && descriptor.request_name.len() <= 512
        && !descriptor.request_name.chars().any(char::is_control)
        && descriptor.version > 0
        && descriptor.length <= max_bytes as u64
        && descriptor.sha256.len() == 64
        && descriptor.sha256 == descriptor.sha256.to_ascii_lowercase()
        && hex::decode(&descriptor.sha256).is_ok_and(|bytes| bytes.len() == 32)
}

fn valid_repository(value: &str) -> bool {
    value.len() <= 512
        && url::Url::parse(value).is_ok_and(|url| {
            url.scheme() == "https"
                && url.query().is_none()
                && url.fragment().is_none()
                && url.path().ends_with('/')
                && url.as_str() == value
        })
}

fn valid_channel(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn canonical_timestamp(value: &str) -> bool {
    value
        .parse::<jiff::Timestamp>()
        .is_ok_and(|timestamp| timestamp.to_string() == value)
}

fn predecessor_shape(sequence: u64, predecessor: Option<&str>) -> bool {
    match (sequence, predecessor) {
        (1, None) => true,
        (1, Some(_)) | (_, None) => false,
        (_, Some(value)) => valid_digest(value),
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value == value.to_ascii_lowercase()
        && hex::decode(value).is_ok_and(|bytes| bytes.len() == 32)
}

fn root_request_matches(descriptor: &RoleDescriptorV0) -> bool {
    descriptor.request_name == format!("{}.root.json", descriptor.version)
}

fn lower_request_matches(descriptor: &RoleDescriptorV0, suffix: &str) -> bool {
    descriptor.request_name == suffix
        || descriptor.request_name == format!("{}.{}", descriptor.version, suffix)
}

fn validate_descriptor_bytes(
    descriptor: &RoleDescriptorV0,
    bytes: &[u8],
) -> Result<(), JournalError> {
    if descriptor.length != bytes.len() as u64 || descriptor.sha256 != hex::encode(sha256(bytes)) {
        return Err(JournalError::Invalid(
            "stored metadata bytes do not match the generation receipt",
        ));
    }
    Ok(())
}
