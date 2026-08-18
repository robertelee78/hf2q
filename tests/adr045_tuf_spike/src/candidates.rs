//! Independent candidate-verifier adapters used only by the spike.

use jiff::Timestamp;
use tempfile::TempDir;
use tough::schema::{Role, Root, Signed, Snapshot, Targets, Timestamp as TufTimestamp};
use tough::{ExpirationEnforcement, Limits, RepositoryLoader};
use url::Url;

use crate::capture_transport::{CapturingTransport, FetchOutcome, FetchRecord};
use crate::model::{
    check_floor, sha256, CapturedRole, CommittedGeneration, RoleKind, SpikeError, MAX_ROOT_BYTES,
    MAX_SNAPSHOT_BYTES, MAX_TARGETS_BYTES, MAX_TIMESTAMP_BYTES,
};
use crate::strict_json;

#[derive(Clone, Debug)]
pub(crate) struct AttemptConfig {
    metadata_base: Url,
    targets_base: Url,
    channel: String,
}

impl AttemptConfig {
    pub(crate) fn new(
        metadata_base: Url,
        targets_base: Url,
        channel: &str,
    ) -> Result<Self, SpikeError> {
        let channel_segment = format!("/{channel}/");
        if !matches!(metadata_base.scheme(), "https")
            || !matches!(targets_base.scheme(), "https")
            || metadata_base.query().is_some()
            || metadata_base.fragment().is_some()
            || targets_base.query().is_some()
            || targets_base.fragment().is_some()
            || !metadata_base.path().ends_with('/')
            || !targets_base.path().ends_with('/')
            || !metadata_base.path().contains(&channel_segment)
            || !targets_base.path().contains(&channel_segment)
            || channel.is_empty()
            || channel.len() > 64
            || !channel
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return Err(SpikeError::TransportPolicy);
        }
        Ok(Self {
            metadata_base,
            targets_base,
            channel: channel.to_string(),
        })
    }

    pub(crate) fn metadata_base(&self) -> &Url {
        &self.metadata_base
    }
}

#[derive(Clone, Debug)]
pub(crate) struct UntrustedRole {
    pub(crate) request_name: String,
    pub(crate) raw: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct AttemptMetadata {
    clock: ClockSample,
    pub(crate) root_chain: Vec<UntrustedRole>,
    pub(crate) timestamp: UntrustedRole,
    pub(crate) snapshot: UntrustedRole,
    pub(crate) targets: UntrustedRole,
}

#[derive(Clone, Copy, Debug)]
struct ClockSample(Timestamp);

/// Opaque result returned only after one candidate verifier and all hf2q-owned
/// floor, expiry, top-level-only, correlation, and repository-policy checks
/// accept the exact raw metadata bytes.
#[derive(Clone, Debug)]
pub(crate) struct VerifiedMetadataEvidence {
    repository: String,
    channel: String,
    update_start: Timestamp,
    verifier_sample: Timestamp,
    prior_root: CapturedRole,
    root_chain: Vec<CapturedRole>,
    root: CapturedRole,
    timestamp: CapturedRole,
    snapshot: CapturedRole,
    targets: CapturedRole,
}

impl VerifiedMetadataEvidence {
    pub(crate) fn repository(&self) -> &str {
        &self.repository
    }

    pub(crate) fn channel(&self) -> &str {
        &self.channel
    }

    pub(crate) fn update_start(&self) -> Timestamp {
        self.update_start
    }

    pub(crate) fn verifier_sample(&self) -> Timestamp {
        self.verifier_sample
    }

    pub(crate) fn root_chain(&self) -> &[CapturedRole] {
        &self.root_chain
    }

    pub(crate) fn prior_root(&self) -> &CapturedRole {
        &self.prior_root
    }

    pub(crate) fn root(&self) -> &CapturedRole {
        &self.root
    }

    pub(crate) fn timestamp(&self) -> &CapturedRole {
        &self.timestamp
    }

    pub(crate) fn snapshot(&self) -> &CapturedRole {
        &self.snapshot
    }

    pub(crate) fn targets(&self) -> &CapturedRole {
        &self.targets
    }

    pub(crate) fn targets_raw(&self) -> &[u8] {
        &self.targets.raw
    }

    pub(crate) fn as_untrusted_attempt(&self) -> AttemptMetadata {
        AttemptMetadata {
            clock: ClockSample(self.update_start),
            root_chain: self
                .root_chain
                .iter()
                .map(|role| UntrustedRole {
                    request_name: role.request_name.clone(),
                    raw: role.raw.clone(),
                })
                .collect(),
            timestamp: untrusted(&self.timestamp),
            snapshot: untrusted(&self.snapshot),
            targets: untrusted(&self.targets),
        }
    }

    pub(crate) fn exact_metadata_eq(&self, other: &Self) -> bool {
        self.repository == other.repository
            && self.channel == other.channel
            && self.update_start == other.update_start
            && self.prior_root == other.prior_root
            && self.root_chain == other.root_chain
            && self.root == other.root
            && self.timestamp == other.timestamp
            && self.snapshot == other.snapshot
            && self.targets == other.targets
    }

    #[cfg(test)]
    pub(crate) fn test_only(
        repository: &str,
        channel: &str,
        update_start: Timestamp,
        prior_root: CapturedRole,
        root_chain: Vec<CapturedRole>,
        roles: [CapturedRole; 4],
    ) -> Self {
        let [root, timestamp, snapshot, targets] = roles;
        Self {
            repository: repository.to_string(),
            channel: channel.to_string(),
            update_start,
            verifier_sample: update_start,
            prior_root,
            root_chain,
            root,
            timestamp,
            snapshot,
            targets,
        }
    }
}

pub(crate) async fn verify_tough_attempt(
    config: &AttemptConfig,
    committed: &CommittedGeneration,
    transport: CapturingTransport,
) -> Result<VerifiedMetadataEvidence, SpikeError> {
    validate_committed(committed)?;
    if transport.metadata_base() != &config.metadata_base {
        return Err(SpikeError::TransportPolicy);
    }
    let update_start = Timestamp::now();
    if update_start < committed.update_start_floor {
        return Err(SpikeError::ClockRollback);
    }
    let scratch = TempDir::new()?;
    tokio::fs::write(
        scratch.path().join("timestamp.json"),
        &committed.raw_timestamp,
    )
    .await?;
    tokio::fs::write(
        scratch.path().join("snapshot.json"),
        &committed.raw_snapshot,
    )
    .await?;
    tokio::fs::write(scratch.path().join("targets.json"), &committed.raw_targets).await?;
    let floor_json = serde_json::to_vec(&committed.update_start_floor)
        .map_err(|_| SpikeError::MalformedMetadata)?;
    tokio::fs::write(scratch.path().join("latest_known_time.json"), floor_json).await?;

    let limits = Limits {
        max_root_size: MAX_ROOT_BYTES as u64,
        max_timestamp_size: MAX_TIMESTAMP_BYTES as u64,
        max_snapshot_size: MAX_SNAPSHOT_BYTES as u64,
        max_targets_size: MAX_TARGETS_BYTES as u64,
        max_root_updates: 32,
    };
    let repository = RepositoryLoader::new(
        &committed.raw_root,
        config.metadata_base.clone(),
        config.targets_base.clone(),
    )
    .transport(transport.clone())
    .limits(limits)
    .datastore(scratch.path())
    .expiration_enforcement(ExpirationEnforcement::Safe)
    .load()
    .await
    .map_err(|_| SpikeError::CandidateRejected)?;
    let update_end = Timestamp::now();
    transport.validate_complete()?;

    let sample_bytes = tokio::fs::read(scratch.path().join("latest_known_time.json")).await?;
    let verifier_sample: Timestamp =
        serde_json::from_slice(&sample_bytes).map_err(|_| SpikeError::InvalidCandidateClock)?;
    if verifier_sample < update_start || verifier_sample > update_end {
        return Err(SpikeError::InvalidCandidateClock);
    }

    let records = transport.records();
    let prior_root = captured(
        &format!("{}.root.json", committed.root.version),
        committed.root.version,
        &committed.raw_root,
    );
    let root_chain = correlate_root_chain(&records)?;
    let root = if let Some(last) = root_chain.last() {
        last.clone()
    } else {
        captured(
            &format!("{}.root.json", repository.root().signed.version.get()),
            repository.root().signed.version.get(),
            &committed.raw_root,
        )
    };
    correlate(repository.root(), &root.raw)?;
    let timestamp = captured_from_record::<TufTimestamp>(
        record_for(&records, RoleKind::Timestamp)?,
        repository.timestamp(),
    )?;
    let snapshot = captured_from_record::<Snapshot>(
        record_for(&records, RoleKind::Snapshot)?,
        repository.snapshot(),
    )?;
    let targets = captured_from_record::<Targets>(
        record_for(&records, RoleKind::Targets)?,
        repository.targets(),
    )?;
    if repository.targets().signed.delegations.is_some() {
        return Err(SpikeError::DelegationsForbidden);
    }
    for expires in [
        repository.root().signed.expires,
        repository.timestamp().signed.expires,
        repository.snapshot().signed.expires,
        repository.targets().signed.expires,
    ] {
        if expires <= verifier_sample {
            return Err(SpikeError::ExpiredAtWrapperTime);
        }
    }
    check_all_floors(committed, &root, &timestamp, &snapshot, &targets)?;

    Ok(VerifiedMetadataEvidence {
        repository: config.metadata_base.to_string(),
        channel: config.channel.clone(),
        update_start: verifier_sample,
        verifier_sample,
        prior_root,
        root_chain,
        root,
        timestamp,
        snapshot,
        targets,
    })
}

pub(crate) fn verify_sigstore_core(
    config: &AttemptConfig,
    committed: &CommittedGeneration,
    attempt: &AttemptMetadata,
) -> Result<VerifiedMetadataEvidence, SpikeError> {
    validate_committed(committed)?;
    let update_start = attempt.clock.0;
    if update_start < committed.update_start_floor {
        return Err(SpikeError::ClockRollback);
    }
    if attempt.root_chain.len() > 32 {
        return Err(SpikeError::MetadataTooLarge);
    }
    let mut trusted = sigstore_tuf::TrustedMetadataSet::from_root(&committed.raw_root)
        .map_err(|_| SpikeError::CandidateRejected)?;
    let prior_root = captured(
        &format!("{}.root.json", committed.root.version),
        committed.root.version,
        &committed.raw_root,
    );
    let mut root_chain = Vec::with_capacity(attempt.root_chain.len());
    for (offset, untrusted_root) in attempt.root_chain.iter().enumerate() {
        strict_json::validate(&untrusted_root.raw, MAX_ROOT_BYTES)?;
        let expected_version = committed
            .root
            .version
            .checked_add(offset as u64 + 1)
            .ok_or(SpikeError::VersionRollback)?;
        if untrusted_root.request_name != format!("{expected_version}.root.json") {
            return Err(SpikeError::TransportPolicy);
        }
        trusted
            .update_root(&untrusted_root.raw)
            .map_err(|_| SpikeError::CandidateRejected)?;
        root_chain.push(parse_captured_root(untrusted_root)?);
    }
    trusted
        .check_root_expired(update_start)
        .map_err(|_| SpikeError::CandidateRejected)?;
    validate_lower_request_names(attempt)?;
    strict_json::validate(&attempt.timestamp.raw, MAX_TIMESTAMP_BYTES)?;
    strict_json::validate(&attempt.snapshot.raw, MAX_SNAPSHOT_BYTES)?;
    strict_json::validate(&attempt.targets.raw, MAX_TARGETS_BYTES)?;
    trusted
        .update_timestamp(&attempt.timestamp.raw, update_start)
        .map_err(|_| SpikeError::CandidateRejected)?;
    trusted
        .update_snapshot(&attempt.snapshot.raw, update_start)
        .map_err(|_| SpikeError::CandidateRejected)?;
    let targets_role = trusted
        .update_targets(&attempt.targets.raw, update_start)
        .map_err(|_| SpikeError::CandidateRejected)?;
    if targets_role.delegations.is_some() {
        return Err(SpikeError::DelegationsForbidden);
    }

    let root = root_chain.last().cloned().unwrap_or_else(|| {
        captured(
            &format!("{}.root.json", committed.root.version),
            committed.root.version,
            &committed.raw_root,
        )
    });
    let timestamp = parse_captured::<TufTimestamp>(&attempt.timestamp)?;
    let snapshot = parse_captured::<Snapshot>(&attempt.snapshot)?;
    let targets = parse_captured::<Targets>(&attempt.targets)?;
    validate_parsed_request_name(&timestamp, "timestamp.json", false)?;
    validate_parsed_request_name(&snapshot, "snapshot.json", true)?;
    validate_parsed_request_name(&targets, "targets.json", true)?;
    check_all_floors(committed, &root, &timestamp, &snapshot, &targets)?;

    Ok(VerifiedMetadataEvidence {
        repository: config.metadata_base.to_string(),
        channel: config.channel.clone(),
        update_start,
        verifier_sample: update_start,
        prior_root,
        root_chain,
        root,
        timestamp,
        snapshot,
        targets,
    })
}

fn validate_lower_request_names(attempt: &AttemptMetadata) -> Result<(), SpikeError> {
    if attempt.timestamp.request_name.len() > 512
        || attempt.snapshot.request_name.len() > 512
        || attempt.targets.request_name.len() > 512
        || attempt.timestamp.request_name != "timestamp.json"
        || !metadata_name_matches(&attempt.snapshot.request_name, "snapshot.json")
        || !metadata_name_matches(&attempt.targets.request_name, "targets.json")
    {
        return Err(SpikeError::TransportPolicy);
    }
    Ok(())
}

fn validate_parsed_request_name(
    role: &CapturedRole,
    suffix: &str,
    allow_versioned: bool,
) -> Result<(), SpikeError> {
    let matches = role.request_name == suffix
        || (allow_versioned && role.request_name == format!("{}.{}", role.version, suffix));
    if !matches {
        return Err(SpikeError::TransportPolicy);
    }
    Ok(())
}

fn check_all_floors(
    committed: &CommittedGeneration,
    root: &CapturedRole,
    timestamp: &CapturedRole,
    snapshot: &CapturedRole,
    targets: &CapturedRole,
) -> Result<(), SpikeError> {
    check_floor(root, &committed.root)?;
    check_floor(timestamp, &committed.timestamp)?;
    check_floor(snapshot, &committed.snapshot)?;
    check_floor(targets, &committed.targets)
}

fn validate_committed(committed: &CommittedGeneration) -> Result<(), SpikeError> {
    let roles = [
        (RoleKind::Root, &committed.raw_root, &committed.root),
        (
            RoleKind::Timestamp,
            &committed.raw_timestamp,
            &committed.timestamp,
        ),
        (
            RoleKind::Snapshot,
            &committed.raw_snapshot,
            &committed.snapshot,
        ),
        (
            RoleKind::Targets,
            &committed.raw_targets,
            &committed.targets,
        ),
    ];
    for (kind, bytes, floor) in roles {
        strict_json::validate(bytes, kind.max_bytes())?;
        if sha256(bytes) != floor.raw_sha256 {
            return Err(SpikeError::CorrelationMismatch);
        }
        let parsed_version = match kind {
            RoleKind::Root => {
                serde_json::from_slice::<Signed<Root>>(bytes).map(|role| role.signed.version.get())
            }
            RoleKind::Timestamp => serde_json::from_slice::<Signed<TufTimestamp>>(bytes)
                .map(|role| role.signed.version.get()),
            RoleKind::Snapshot => serde_json::from_slice::<Signed<Snapshot>>(bytes)
                .map(|role| role.signed.version.get()),
            RoleKind::Targets => serde_json::from_slice::<Signed<Targets>>(bytes)
                .map(|role| role.signed.version.get()),
        }
        .map_err(|_| SpikeError::MalformedMetadata)?;
        if parsed_version != floor.version {
            return Err(SpikeError::CorrelationMismatch);
        }
    }
    Ok(())
}

fn correlate_root_chain(records: &[FetchRecord]) -> Result<Vec<CapturedRole>, SpikeError> {
    let mut chain = Vec::new();
    for record in records.iter().filter(|record| {
        record.role == Some(RoleKind::Root) && record.outcome == FetchOutcome::Complete
    }) {
        strict_json::validate(&record.bytes, MAX_ROOT_BYTES)?;
        let parsed: Signed<Root> =
            serde_json::from_slice(&record.bytes).map_err(|_| SpikeError::MalformedMetadata)?;
        chain.push(captured(
            &record.request_name,
            parsed.signed.version.get(),
            &record.bytes,
        ));
    }
    Ok(chain)
}

fn record_for(records: &[FetchRecord], kind: RoleKind) -> Result<&FetchRecord, SpikeError> {
    let mut matching = records
        .iter()
        .filter(|record| record.role == Some(kind) && record.outcome == FetchOutcome::Complete);
    let record = matching.next().ok_or(SpikeError::TransportPolicy)?;
    if matching.next().is_some() {
        return Err(SpikeError::TransportPolicy);
    }
    Ok(record)
}

fn captured_from_record<R>(
    record: &FetchRecord,
    verified: &Signed<R>,
) -> Result<CapturedRole, SpikeError>
where
    R: Role + serde::de::DeserializeOwned + PartialEq,
{
    strict_json::validate(
        &record.bytes,
        record.role.expect("matched role").max_bytes(),
    )?;
    correlate(verified, &record.bytes)?;
    Ok(captured(
        &record.request_name,
        verified.signed.version().get(),
        &record.bytes,
    ))
}

fn correlate<R>(verified: &Signed<R>, raw: &[u8]) -> Result<(), SpikeError>
where
    R: serde::de::DeserializeOwned + PartialEq,
{
    let parsed: Signed<R> =
        serde_json::from_slice(raw).map_err(|_| SpikeError::MalformedMetadata)?;
    if &parsed != verified {
        return Err(SpikeError::CorrelationMismatch);
    }
    Ok(())
}

fn parse_captured_root(role: &UntrustedRole) -> Result<CapturedRole, SpikeError> {
    parse_captured::<Root>(role)
}

fn parse_captured<R>(role: &UntrustedRole) -> Result<CapturedRole, SpikeError>
where
    R: Role + serde::de::DeserializeOwned,
{
    let parsed: Signed<R> =
        serde_json::from_slice(&role.raw).map_err(|_| SpikeError::MalformedMetadata)?;
    Ok(captured(
        &role.request_name,
        parsed.signed.version().get(),
        &role.raw,
    ))
}

fn captured(request_name: &str, version: u64, raw: &[u8]) -> CapturedRole {
    CapturedRole {
        request_name: request_name.to_string(),
        version,
        raw: raw.to_vec(),
        raw_sha256: sha256(raw),
    }
}

fn untrusted(role: &CapturedRole) -> UntrustedRole {
    UntrustedRole {
        request_name: role.request_name.clone(),
        raw: role.raw.clone(),
    }
}

fn metadata_name_matches(name: &str, suffix: &str) -> bool {
    name == suffix
        || name
            .strip_suffix(suffix)
            .and_then(|prefix| prefix.strip_suffix('.'))
            .is_some_and(|version| {
                !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
            })
}

pub(crate) fn attempt_from_fixture(
    fixture: &crate::test_repository::RepositoryFixture,
) -> AttemptMetadata {
    AttemptMetadata {
        clock: ClockSample(Timestamp::now()),
        root_chain: Vec::new(),
        timestamp: UntrustedRole {
            request_name: "timestamp.json".to_string(),
            raw: fixture.timestamp.clone(),
        },
        snapshot: UntrustedRole {
            request_name: "snapshot.json".to_string(),
            raw: fixture.snapshot.clone(),
        },
        targets: UntrustedRole {
            request_name: "targets.json".to_string(),
            raw: fixture.targets.clone(),
        },
    }
}

pub(crate) fn committed_from_fixture(
    fixture: &crate::test_repository::RepositoryFixture,
    version: u64,
) -> CommittedGeneration {
    let root: Signed<Root> = serde_json::from_slice(&fixture.root).expect("fixture root parses");
    CommittedGeneration {
        update_start_floor: Timestamp::now(),
        root: crate::model::RoleFloor::from_bytes(root.signed.version.get(), &fixture.root),
        timestamp: crate::model::RoleFloor::from_bytes(version, &fixture.timestamp),
        snapshot: crate::model::RoleFloor::from_bytes(version, &fixture.snapshot),
        targets: crate::model::RoleFloor::from_bytes(version, &fixture.targets),
        raw_root: fixture.root.clone(),
        raw_timestamp: fixture.timestamp.clone(),
        raw_snapshot: fixture.snapshot.clone(),
        raw_targets: fixture.targets.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{attempt_from_fixture, correlate, TufTimestamp};
    use crate::model::SpikeError;
    use jiff::Timestamp;
    use tough::schema::Signed;

    #[test]
    fn raw_correlation_compares_the_complete_parsed_role_not_only_version() {
        let raw = include_bytes!("../testdata/timestamp-v2-normalized.json");
        let verified: Signed<TufTimestamp> = serde_json::from_slice(raw).unwrap();
        let semantically_equal = serde_json::to_vec(&verified).unwrap();
        assert_ne!(semantically_equal, raw);
        correlate(&verified, &semantically_equal).expect("equivalent envelope correlates");

        let mut changed: serde_json::Value = serde_json::from_slice(raw).unwrap();
        changed["signatures"][0]["sig"] = serde_json::Value::String("00".repeat(64));
        assert!(matches!(
            correlate(&verified, &serde_json::to_vec(&changed).unwrap()),
            Err(SpikeError::CorrelationMismatch)
        ));
    }

    #[tokio::test]
    async fn untrusted_fixture_attempt_samples_its_clock_inside_the_adapter() {
        let fixture = crate::test_repository::build_repository(1)
            .await
            .expect("fixture builds");
        let before = Timestamp::now();
        let attempt = attempt_from_fixture(&fixture);
        let after = Timestamp::now();
        assert!(attempt.clock.0 >= before);
        assert!(attempt.clock.0 <= after);
    }
}
