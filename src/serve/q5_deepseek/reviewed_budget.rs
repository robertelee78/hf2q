//! Private quantitative operating validator. A cap is not a completion promise.
use super::{
    operating_case::{self, Limits},
    operating_os,
    packet::Packet,
    startup,
};
use anyhow::{ensure, Context, Result};
use mlx_native::{gguf::GgufFile, LiveRootLimits, RootCreationLimits};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Policy {
    pub page_bytes: u64,
    pub max_buffer_bytes: u64,
    pub device_registry_id: u64,
    pub native_live_bytes: u64,
    pub native_live_roots: u64,
    pub native_attempt_bytes: u64,
    pub native_attempt_roots: u64,
    pub native_raw_attempts: u64,
    pub rope_entries: u64,
    pub cpu_bytes: u64,
    pub tools_bytes: u64,
    pub client_bytes: u64,
    pub driver_bytes: u64,
    pub observer_bytes: u64,
    pub inactive_file_residual_bytes: u64,
    pub additional_noncase_bytes: u64,
    pub reaction_bytes: u64,
    pub runtime_free_bytes: u64,
    pub deadline_ms: u64,
    pub sample_interval_ms: u64,
    pub max_http_inflight: u32,
    pub process_stop_bytes: u64,
    pub max_decoded_bytes: u64,
    pub count_file_cache_as_operating_available: bool,
    pub charge_creation_envelope: bool,
    pub retired_native_residual_bytes: u64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub policy: Policy,
    pub limits: Limits,
    pub review_path: String,
    pub review_sha256: String,
    // Exact root-reviewed source/artifact/workload refs remain evidence inputs;
    // actual quantities and typed options are independently rederived below.
    pub capability_receipt_path: String,
    pub capability_receipt_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Review {
    schema: String,
    verdict: String,
    source_commit: String,
    hf_source_sha256: String,
    native_version: String,
    binary_sha256: String,
    artifact_sha256: String,
    workload_sha256: String,
    policy_sha256: String,
    case_sha256: String,
    capability_receipt_sha256: String,
    supersedes_operating_policy_sha256: String,
    issued_unix_ms: u64,
    expires_unix_ms: u64,
    residual_rationale: String,
    file: super::packet::FileIdentity,
    native_package_sha256: String,
    physical_admission: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Capability {
    max_buffer_length: u64,
    name: String,
    physical_memory: u64,
    recommended_max_working_set_size: u64,
    registry_id: u64,
    unified_memory: bool,
}
pub(super) struct ReviewedSourceBudget {
    creation: RootCreationLimits,
    live: LiveRootLimits,
    case: Limits,
    policy: Policy,
    swap: u64,
    minimum: crate::inference::models::deepseek4::selected_budget_plan::Minimum,
    initial: operating_os::Sample,
}
impl ReviewedSourceBudget {
    pub(super) fn into_limits(self) -> Result<(RootCreationLimits, LiveRootLimits)> {
        operating_case::install(
            self.case,
            self.policy,
            self.swap,
            self.minimum,
            self.initial,
        )?;
        Ok((self.creation, self.live))
    }
}
fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn hash(s: &str) -> Result<()> {
    ensure!(
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "Q5 canonical reviewed SHA256"
    );
    Ok(())
}
fn evidence<T: serde::de::DeserializeOwned>(path: &str, expected: &str) -> Result<T> {
    use std::io::Read;
    hash(expected)?;
    let p = Path::new(path);
    ensure!(
        p.is_absolute() && path.len() <= 4096,
        "Q5 bounded evidence path"
    );
    let mut b = Vec::new();
    super::diagnostic_open(p)?.take(65537).read_to_end(&mut b)?;
    ensure!(
        b.len() <= 65536 && sha(&b) == expected,
        "Q5 evidence quota/hash"
    );
    Ok(serde_json::from_slice(&b)?)
}
fn case_digest(c: &Case) -> Result<String> {
    // Review refs are outside this digest to avoid a self-hash cycle.
    Ok(sha(&serde_json::to_vec(&(
        c.policy,
        c.limits,
        &c.capability_receipt_sha256,
    ))?))
}
fn numeric(
    p: Policy,
    l: Limits,
    minimum: crate::inference::models::deepseek4::selected_budget_plan::Minimum,
    s: operating_os::Sample,
    collector: u64,
) -> Result<()> {
    ensure!(
        l.queue_capacity > 0
            && l.queue_capacity <= 64
            && l.max_enqueue_attempts > 0
            && l.max_enqueue_attempts <= 1_000_000
            && l.max_prompt_tokens == 262144
            && l.max_output_tokens == 8192,
        "Q5 finite selected case limits"
    );
    ensure!(
        p.max_http_inflight > 0
            && p.max_http_inflight <= 64
            && u64::from(p.max_http_inflight) <= l.max_enqueue_attempts,
        "Q5 finite preparse ingress bound"
    );
    ensure!(
        p.process_stop_bytes > 0
            && p.process_stop_bytes <= s.total
            && s.rss <= p.process_stop_bytes
            && s.footprint <= p.process_stop_bytes
            && p.max_decoded_bytes > 0
            && p.max_decoded_bytes <= p.cpu_bytes,
        "Q5 process/output operating stops"
    );
    ensure!(
        p.sample_interval_ms > 0
            && p.sample_interval_ms <= 1000
            && p.deadline_ms >= p.sample_interval_ms
            && p.deadline_ms <= 3_600_000,
        "Q5 finite cadence/deadline"
    );
    ensure!(
        p.page_bytes == s.page
            && p.max_buffer_bytes >= p.page_bytes
            && p.max_buffer_bytes <= s.total
            && p.max_buffer_bytes % p.page_bytes == 0
            && p.device_registry_id > 0,
        "Q5 geometry/OS mismatch"
    );
    ensure!(
        p.native_live_roots
            >= minimum
                .weight_roots
                .checked_add(minimum.cache_roots)
                .context("Q5 mandatory root sum overflow")? as u64
            && p.native_live_roots <= 1_000_000
            && p.native_attempt_roots >= p.native_live_roots
            && p.native_attempt_roots <= 1_000_000
            && p.native_raw_attempts >= p.native_attempt_roots
            && p.native_raw_attempts <= 1_000_000
            && p.rope_entries > 0
            && p.rope_entries <= 1_000_000,
        "Q5 bounded native root/history quotas"
    );
    ensure!(
        p.native_attempt_bytes
            <= p.native_attempt_roots
                .checked_mul(p.max_buffer_bytes)
                .context("Q5 historical byte product overflow")?,
        "Q5 historical bytes exceed finite root/maximum-buffer product"
    );
    let mandatory = minimum
        .weights
        .checked_add(minimum.four_caches_stub)
        .context("Q5 mandatory native sum overflow")?;
    ensure!(
        p.native_live_bytes >= mandatory && p.native_attempt_bytes >= p.native_live_bytes,
        "Q5 native cap below mandatory fully charged capacity"
    );
    let positives = [
        p.cpu_bytes,
        p.tools_bytes,
        p.client_bytes,
        p.driver_bytes,
        p.observer_bytes,
        p.inactive_file_residual_bytes,
        p.retired_native_residual_bytes,
        p.reaction_bytes,
        p.runtime_free_bytes,
    ];
    ensure!(
        positives.iter().all(|&x| x > 0) && p.observer_bytes >= collector,
        "Q5 explicit positive cells/collector charge"
    );
    let cpu_wire = (l.queue_capacity as u64)
        .checked_mul(262144 * 4)
        .and_then(|x| x.checked_add(u64::from(p.max_http_inflight) * 8 * 1024 * 1024))
        .context("Q5 queue/ingress bound overflow")?;
    ensure!(
        p.cpu_bytes >= cpu_wire,
        "Q5 CPU allowance below admitted raw prompt/ingress capacities"
    );
    let charged_native = if p.charge_creation_envelope {
        p.native_attempt_bytes
    } else {
        p.native_live_bytes
    };
    let total = positives.iter().try_fold(
        charged_native
            .checked_add(p.additional_noncase_bytes)
            .context("Q5 noncase overflow")?,
        |n, &x| n.checked_add(x).context("Q5 positive sum overflow"),
    )?;
    let working = s
        .free
        .checked_add(if p.count_file_cache_as_operating_available {
            s.file_cache
        } else {
            0
        })
        .context("Q5 working estimate overflow")?;
    ensure!(
        working <= s.total && total <= working && s.free >= p.runtime_free_bytes,
        "Q5 fully charged case exceeds fresh Free+file-cache WORKING estimate; no live-root refund"
    );
    Ok(())
}
fn inspect(
    packet: &Packet,
    g: &GgufFile,
    facts: &startup::Facts,
    actual_case: bool,
) -> Result<(RootCreationLimits, LiveRootLimits, Limits, Policy, u64, u64)> {
    closed_environment()?;
    let begin = Instant::now();
    let c = packet
        .operating_case
        .as_ref()
        .context("Q5 operating case missing")?;
    ensure!(
        matches!(facts, startup::Facts::BoundedEmbeddedMetadata { .. }),
        "Q5 bounded actual startup required"
    );
    let r: Review = evidence(&c.review_path, &c.review_sha256)?;
    let cap: Capability = evidence(&c.capability_receipt_path, &c.capability_receipt_sha256)?;
    let policy_hash = sha(&serde_json::to_vec(&c.policy)?);
    hash(&r.supersedes_operating_policy_sha256)?;
    ensure!(
        !r.residual_rationale.is_empty() && r.residual_rationale.len() <= 1024,
        "Q5 explicit independently reviewed retirement/OS residual rationale required"
    );
    ensure!(
        !r.physical_admission
            && r.schema == "hf2q.q5.deepseek.operating-case-review.v1"
            && r.verdict == "AcceptedForOneGuardedAttempt"
            && r.source_commit == option_env!("HF2Q_BUILD_GIT_SHA").unwrap_or("Unknown")
            && r.native_version == env!("HF2Q_MLX_NATIVE_VERSION")
            && r.file == packet.file
            && r.native_package_sha256 == packet.native_package_sha256
            && r.native_package_sha256
                == "bb508d36e311a24147e917282a8ccae0d47720f4a12ae678717df5d569c75e3c"
            && r.hf_source_sha256 == packet.hf_source_sha256
            && r.binary_sha256 == packet.binary_sha256
            && r.artifact_sha256 == packet.artifact_sha256
            && r.workload_sha256 == packet.workload_sha256
            && r.policy_sha256 == policy_hash
            && packet.policy_sha256 == policy_hash
            && r.case_sha256 == case_digest(c)?
            && r.capability_receipt_sha256 == c.capability_receipt_sha256,
        "Q5 exact reviewed source/native/binary/artifact/workload/policy/case joins"
    );
    let now = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())?;
    review_fresh(r.issued_unix_ms, r.expires_unix_ms, now)?;
    let s = operating_os::sample()?;
    ensure!(
        cap.unified_memory
            && !cap.name.is_empty()
            && cap.physical_memory == s.total
            && cap.registry_id == c.policy.device_registry_id
            && cap.max_buffer_length == c.policy.max_buffer_bytes
            && cap.recommended_max_working_set_size <= s.total,
        "Q5 actual capability receipt mismatch"
    );
    let minimum = crate::inference::models::deepseek4::selected_budget_plan::derive(
        g,
        usize::try_from(c.policy.page_bytes)?,
        usize::try_from(c.policy.max_buffer_bytes)?,
        packet.file.bytes,
    )?;
    let decoded = startup::decoded_token_bound(g)?
        .checked_mul(c.limits.max_output_tokens as u64)
        .context("Q5 output bound overflow")?;
    ensure!(
        decoded <= c.policy.max_decoded_bytes,
        "Q5 maximum actual token/output bytes exceed reviewed output ceiling"
    );
    numeric(c.policy, c.limits, minimum, s, packet.collector_upper_bytes)?;
    if actual_case {
        operating_case::require_actual(c.limits)?;
    }
    ensure!(
        begin.elapsed().as_millis() <= u128::from(c.policy.sample_interval_ms),
        "Q5 validation work exceeded reaction freshness bound"
    );
    Ok((
        RootCreationLimits {
            max_attempted_requested_bytes: usize::try_from(c.policy.native_attempt_bytes)?,
            max_root_attempts: usize::try_from(c.policy.native_attempt_roots)?,
            max_raw_constructor_attempts: usize::try_from(c.policy.native_raw_attempts)?,
            max_registered_devices: 1,
            max_rope_cache_entries: usize::try_from(c.policy.rope_entries)?,
        },
        LiveRootLimits {
            max_live_and_pending_requested_bytes: usize::try_from(c.policy.native_live_bytes)?,
            max_live_and_pending_roots: usize::try_from(c.policy.native_live_roots)?,
        },
        c.limits,
        c.policy,
        s.swap,
        r.expires_unix_ms,
    ))
}
pub(super) fn dry_validate(p: &Packet, g: &GgufFile, f: &startup::Facts) -> Result<()> {
    inspect(p, g, f, false).map(|_| ())
}
pub(super) fn validate(
    p: &Packet,
    g: &GgufFile,
    f: &startup::Facts,
) -> Result<ReviewedSourceBudget> {
    let (creation, live, case, policy, swap, expires) = inspect(p, g, f, true)?;
    // Same delivered executable is independently streamed, bounded, no model read.
    use std::io::Read;
    let mut file = std::fs::File::open(std::env::current_exe()?)?;
    ensure!(
        file.metadata()?.len() <= 256 * 1024 * 1024,
        "Q5 executable identity bound"
    );
    let begin = Instant::now();
    let mut h = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
        ensure!(
            begin.elapsed().as_millis() <= u128::from(policy.sample_interval_ms),
            "Q5 executable hash freshness"
        );
    }
    ensure!(
        format!("{:x}", h.finalize()) == p.binary_sha256,
        "Q5 actual running executable mismatch"
    );
    let now = operating_os::sample()?;
    let minimum = crate::inference::models::deepseek4::selected_budget_plan::derive(
        g,
        policy.page_bytes as usize,
        policy.max_buffer_bytes as usize,
        p.file.bytes,
    )?;
    numeric(policy, case, minimum, now, p.collector_upper_bytes)?;
    ensure!(now.swap <= swap, "Q5 swap grew before native installation");
    ensure!(
        u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())? <= expires
            && begin.elapsed().as_millis() <= u128::from(policy.sample_interval_ms),
        "Q5 final identity/OS END freshness"
    );
    Ok(ReviewedSourceBudget {
        creation,
        live,
        case,
        policy,
        swap,
        minimum,
        initial: now,
    })
}
fn review_fresh(issued: u64, expires: u64, now: u64) -> Result<()> {
    ensure!(
        issued <= now
            && now <= expires
            && expires.checked_sub(issued).is_some_and(|x| x <= 86_400_000),
        "Q5 stale/future operating review"
    );
    Ok(())
}
/// Selected-only closed activation set. No unknown dump/trace/CPU-writing
/// coactivation may hide outside this operating case. None never calls this.
fn closed_environment() -> Result<()> {
    let mut count = 0usize;
    let mut bytes = 0usize;
    for (k, v) in std::env::vars_os() {
        let k = k.to_str().context("Q5 environment key UTF8")?;
        if k.starts_with("HF2Q_")
            || k.starts_with("MLX_")
            || k.starts_with("MTL_")
            || k.starts_with("METAL_")
            || k.starts_with("DYLD_")
        {
            count += 1;
            bytes = bytes
                .checked_add(k.len())
                .and_then(|n| n.checked_add(v.len()))
                .context("Q5 environment byte overflow")?;
            ensure!(
                count <= 16 && bytes <= 65536,
                "Q5 bounded selected environment"
            );
            ensure!(
                match k {
                    "HF2Q_DEEPSEEK_OWNER_PACKET" => v
                        .to_str()
                        .is_some_and(|s| s.starts_with('/') && s.len() <= 4096),
                    "HF2Q_UNSAFE_EXPERIMENTS"
                    | "HF2Q_DEEPSEEK_MMAP_WEIGHTS"
                    | "HF2Q_ENCODER_SESSION" => v == "1",
                    _ => false,
                },
                "Q5 unsupported selected coactivation {k}"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "reviewed_budget_tests.rs"]
mod tests;
