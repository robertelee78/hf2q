//! ADR-063 F2 bounded metadata observation. Default observation never schedules.
//! An explicit diagnostic-only demand config selects its pre-admission quantum.
//! Unknown execution/readiness is explicit and cannot qualify a measured mix.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
mod metadata_budget;
mod demand;
mod native;

pub(crate) const SCHEMA: &str = "hf2q.f2.phase-trace.v1";
const EVENT_LIMIT: usize = 64 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    output: PathBuf,
    run_id: String,
    server_budget_bytes: u64,
    client_budget_bytes: u64,
    #[serde(default, deserialize_with = "metadata_budget::deserialize_optional")]
    metadata_budget_profile: Option<metadata_budget::Profile>,
    #[serde(default)]
    native_observer: Option<native::Config>,
    #[serde(default, deserialize_with = "demand::deserialize_optional")]
    forward_demand_policy: Option<demand::Config>,
    #[serde(deserialize_with = "unique_identity")]
    identity: BTreeMap<String, String>,
}

fn unique_identity<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<String, String>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("eight unique source identity strings")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((key, value)) = a.next_entry::<String, String>()? {
                if result.insert(key, value).is_some() || result.len() > 8 {
                    return Err(serde::de::Error::custom(
                        "duplicate/oversized trace identity",
                    ));
                }
            }
            Ok(result)
        }
    }
    d.deserialize_map(Unique)
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SlotState {
    pub request_id: u64,
    pub slot_id: u32,
    pub generation: u64,
    pub phase: &'static str,
    pub active: bool,
    pub forward_eligibility: Option<bool>,
    pub runnable_forward: Option<bool>,
    pub runnable_emit: Option<bool>,
    pub inflight: Option<bool>,
    pub blocked: Option<&'static str>,
}

/// An owning-worker fact, never inferred from a scheduler phase or row width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkState {
    pub handle: crate::serve::scheduler::SlotHandle,
    pub forward_eligibility: Option<bool>,
    pub runnable_forward: Option<bool>,
    pub runnable_emit: Option<bool>,
    pub blocked: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PhysicalRow {
    pub request_id: u64,
    pub slot_id: u32,
    pub generation: u64,
    pub token_position: u64,
    /// Contiguous logical token positions; avoids per-token metadata allocation.
    pub token_count: u32,
    pub phase: &'static str,
}

struct Recorder {
    file: File,
    origin_ns: u64,
    sealed: bool,
    workers: usize,
    sequence: u64,
    bytes: u64,
    budget: u64,
    invalid: Option<String>,
    config_sha256: String,
    native_config: Option<native::Config>,
    demand_config: Option<demand::Config>,
}

fn checked_charge(bytes: u64, length: usize, budget: u64) -> Option<u64> {
    if length > EVENT_LIMIT {
        return None;
    }
    bytes
        .checked_add(u64::try_from(length).ok()?.checked_add(1)?)
        .filter(|n| *n <= budget)
}

/// One JSONL record with keys in `schema, seq, at_ns, kind, payload` order;
/// a typed payload serializes directly, without an intermediate `Value`.
#[derive(Serialize)]
struct Line<'a, P> {
    schema: &'static str,
    seq: u64,
    at_ns: u64,
    kind: &'a str,
    payload: P,
}

impl Recorder {
    fn append(&mut self, kind: &str, payload: Value) {
        self.append_all(kind, std::iter::once(payload));
    }

    /// Same-kind records in order with one file write. Each record is still
    /// sequenced and charged on its own, and `at_ns` is taken as it is
    /// serialized, up to one batch before the write. A record that fails to
    /// serialize or charge ends the batch with the earlier records written,
    /// as single appends would. A failed write leaves the sequence and byte
    /// counts unadvanced for the whole batch, though part of it may be in the
    /// file. Every failure is sticky, so an invalid trace gets no `end` record.
    fn append_all<P: Serialize>(&mut self, kind: &str, payloads: impl IntoIterator<Item = P>) {
        if self.invalid.is_some() {
            return;
        }
        if self.sealed {
            self.invalid = Some("event after trace finalization".into());
            return;
        }
        let (mut sequence, mut bytes) = (self.sequence, self.bytes);
        let mut lines = Vec::new();
        let mut failure = None;
        for payload in payloads {
            let start = lines.len();
            let result = (|| -> Result<(u64, u64), String> {
                let next = sequence.checked_add(1).ok_or("sequence overflow")?;
                let ns = monotonic_ns()?
                    .checked_sub(self.origin_ns)
                    .ok_or("monotonic clock regressed")?;
                let line = Line {
                    schema: SCHEMA,
                    seq: sequence,
                    at_ns: ns,
                    kind,
                    payload,
                };
                serde_json::to_writer(&mut lines, &line).map_err(|e| e.to_string())?;
                let charged = checked_charge(bytes, lines.len() - start, self.budget)
                    .ok_or("trace byte/event quota exceeded; no trimming or qualification")?;
                lines.push(b'\n');
                Ok((next, charged))
            })();
            match result {
                Ok(next) => (sequence, bytes) = next,
                Err(error) => {
                    lines.truncate(start);
                    failure = Some(error);
                    break;
                }
            }
        }
        if !lines.is_empty() {
            if let Err(error) = self.file.write_all(&lines).and_then(|_| self.file.flush()) {
                self.invalid = Some(error.to_string());
                return;
            }
            self.sequence = sequence;
            self.bytes = bytes;
        }
        if failure.is_some() {
            self.invalid = failure;
        }
    }
}

enum Trace {
    Disabled,
    Invalid(String),
    Active(Mutex<Recorder>),
}
static TRACE: OnceLock<Trace> = OnceLock::new();

fn validate_config(c: &Config) -> Result<(), String> {
    if let Some(policy) = &c.forward_demand_policy {
        policy.validate(c.native_observer.as_ref().map(|n| n.diagnostic_case))?;
        if !c.native_observer.as_ref().is_some_and(|n| n.witness_manifest.input_sources.iter()
            .all(|source| matches!(source.as_str(), "qwen35.forward.actual_tokens" | "qwen35.greedy.actual_tokens"))) {
            return Err("queued-demand diagnostic requires Qwen actual-input witnesses".into());
        }
    }
    let total_limit = metadata_budget::limit(
        c.metadata_budget_profile,
        c.native_observer
            .as_ref()
            .map(|native| native.diagnostic_case),
    )?;
    if c.schema != SCHEMA
        || c.server_budget_bytes == 0
        || c.client_budget_bytes < 128 * 1024 * 1024
        || c.server_budget_bytes
            .checked_add(c.client_budget_bytes)
            .filter(|n| *n <= total_limit)
            .is_none()
    {
        return Err("invalid combined server/client trace quota".into());
    }
    if let Some(native) = &c.native_observer {
        native.validate(
            c.server_budget_bytes,
            metadata_budget::event_limit(c.metadata_budget_profile),
        )?;
    }
    let keys = [
        "server_source",
        "server_binary_sha256",
        "native_version",
        "native_checksum",
        "model_sha256",
        "tokenizer_sha256",
        "template_sha256",
        "profile_sha256",
    ];
    if c.run_id.is_empty()
        || c.run_id.len() > 128
        || !c
            .run_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("bounded run identity required".into());
    }
    if c.identity.len() != keys.len() || keys.iter().any(|k| !c.identity.contains_key(*k)) {
        return Err("closed source-bound trace identity required".into());
    }
    for (key, value) in &c.identity {
        let valid = if key == "native_version" {
            !value.is_empty()
                && value.len() <= 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-+".contains(&b))
        } else {
            value.len() == (if key == "server_source" { 40 } else { 64 })
                && value
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if !valid {
            return Err(format!("invalid trace identity {key}"));
        }
    }
    let parent = c.output.parent().ok_or("trace output parent")?;
    if !c.output.is_absolute()
        || parent.canonicalize().map_err(|e| e.to_string())? != parent
        || c.output.try_exists().map_err(|e| e.to_string())?
        || parent
            .ancestors()
            .any(|p| p.join(".git").symlink_metadata().is_ok())
    {
        return Err("trace output must be new, private and outside Git ancestors".into());
    }
    Ok(())
}

pub(crate) fn monotonic_ns() -> Result<u64, String> {
    let mut t = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: valid writable timespec; no payload or device access.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, t.as_mut_ptr()) } != 0 {
        return Err("CLOCK_MONOTONIC unavailable".into());
    }
    let t = unsafe { t.assume_init() };
    u64::try_from(t.tv_sec)
        .ok()
        .and_then(|s| s.checked_mul(1_000_000_000))
        .and_then(|s| {
            u64::try_from(t.tv_nsec)
                .ok()
                .filter(|n| *n < 1_000_000_000)
                .and_then(|n| s.checked_add(n))
        })
        .ok_or("monotonic clock overflow".into())
}

fn initialize() -> Trace {
    let Some(path) = std::env::var_os("HF2Q_PHASE_TRACE_CONFIG") else {
        return Trace::Disabled;
    };
    let setup = (|| -> Result<Recorder, String> {
        let path = Path::new(&path);
        if !path.is_absolute() {
            return Err("bounded absolute trace config required".into());
        }
        let file = File::open(path).map_err(|e| e.to_string())?;
        if !file.metadata().map_err(|e| e.to_string())?.is_file() {
            return Err("regular trace configuration file required".into());
        }
        let mut bytes = Vec::with_capacity(16385);
        file.take(16385)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 16384 {
            return Err("trace configuration quota exceeded".into());
        }
        let c: Config = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        validate_config(&c)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&c.output)
            .map_err(|e| e.to_string())?;
        let mut recorder = Recorder {
            file,
            origin_ns: monotonic_ns()?,
            sealed: false,
            workers: 0,
            sequence: 0,
            bytes: 0,
            budget: c.server_budget_bytes
                - c.native_observer
                    .as_ref()
                    .map_or(0, |n| n.reservation_bytes()),
            invalid: None,
            config_sha256: hex::encode(Sha256::digest(&bytes)),
            native_config: c.native_observer.clone(),
            demand_config: c.forward_demand_policy.clone(),
        };
        let mut header = json!({"identity":c.identity,"run_id":c.run_id,
            "server_budget_bytes":c.server_budget_bytes,"client_budget_bytes":c.client_budget_bytes,
            "clock":"CLOCK_MONOTONIC nanoseconds since origin","clock_id":"CLOCK_MONOTONIC","origin_monotonic_ns":recorder.origin_ns,
            "config_sha256":recorder.config_sha256});
        if let Some(policy) = c.forward_demand_policy {
            header["forward_demand_policy"] = json!(policy);
        }
        if let Some(profile) = c.metadata_budget_profile {
            header["metadata_budget_profile"] = json!(profile);
        }
        if let Some(native) = c.native_observer {
            header["native_observer"] = json!(native);
            header["server_jsonl_budget_bytes"] = json!(recorder.budget);
        }
        recorder.append("header", header);
        Ok(recorder)
    })();
    match setup {
        Ok(r) => Trace::Active(Mutex::new(r)),
        Err(e) => Trace::Invalid(e),
    }
}

pub(crate) fn enabled() -> bool {
    !matches!(TRACE.get_or_init(initialize), Trace::Disabled)
}

/// Called before any device/model creation. Invalid opt-in configuration refuses
/// before a costly load; default absence does not install a native observer.
pub(crate) fn initialize_early() -> Result<(), String> {
    match TRACE.get_or_init(initialize) {
        Trace::Disabled => Ok(()),
        Trace::Invalid(error) => Err(error.clone()),
        Trace::Active(lock) => {
            let config = lock
                .lock()
                .map_err(|_| "trace lock poisoned")?
                .native_config
                .clone();
            if let Some(config) = config {
                native::install(&config)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn native_control_nonce() -> Option<String> {
    match TRACE.get() {
        Some(Trace::Active(lock)) => lock
            .lock()
            .ok()?
            .native_config
            .as_ref()
            .map(|c| c.control_nonce.clone()),
        _ => None,
    }
}

pub(crate) fn invalidate(reason: &'static str) {
    if let Trace::Active(lock) = TRACE.get_or_init(initialize) {
        if let Ok(mut r) = lock.lock() {
            r.invalid = Some(reason.into());
        }
    }
}

pub(crate) fn publish(kind: &str, payload: Value) {
    if let Trace::Active(lock) = TRACE.get_or_init(initialize) {
        if let Ok(mut r) = lock.lock() {
            r.append(kind, payload);
        }
    }
}

/// One trace lock and one file write for an ordered same-kind batch.
fn publish_all<P: Serialize>(kind: &str, payloads: impl IntoIterator<Item = P>) {
    if let Trace::Active(lock) = TRACE.get_or_init(initialize) {
        if let Ok(mut r) = lock.lock() {
            r.append_all(kind, payloads);
        }
    }
}

/// Scheduler phase is not proof of engine readiness or actual GPU submission.
pub(crate) fn scheduler(states: Vec<SlotState>, cause: &'static str) {
    if states.len() > 256 {
        if let Trace::Active(lock) = TRACE.get_or_init(initialize) {
            if let Ok(mut r) = lock.lock() {
                r.invalid = Some("slot metadata bound exceeded".into());
            }
        }
        return;
    }
    crate::inference::native_observation::update_owners(&states);
    publish_slot_states(states, cause, &[]);
}

/// Diagnostic selection happens before admission. No opt-in means no scheduler change.
pub(crate) fn qwen_prefill_quantum(default: u32) -> Result<u32, String> {
    let selected = match TRACE.get_or_init(initialize) {
        Trace::Disabled => return Ok(default),
        Trace::Invalid(error) => return Err(error.clone()),
        Trace::Active(lock) => lock.lock().map_err(|_| "trace lock poisoned")?
            .demand_config.clone(),
    };
    if let Some(policy) = selected {
        Ok(policy.qwen_prefill_chunk_tokens)
    } else {
        Ok(default)
    }
}

/// Called only AFTER the owning Qwen worker installs its actual setter.
pub(crate) fn qwen_quantum_installed(tokens: u32) {
    let selected = match TRACE.get() {
        Some(Trace::Active(lock)) => lock.lock().ok().and_then(|r| r.demand_config.clone()),
        _ => None,
    };
    if let Some(policy) = selected {
        if tokens != policy.qwen_prefill_chunk_tokens {
            invalidate("diagnostic quantum installation mismatch");
            return;
        }
        publish("diagnostic_quantum", json!({"family":"qwen35",
            "population_policy":policy.population_policy,
            "qwen_prefill_chunk_tokens":tokens,"installed_before_admission":true}));
    }
}

pub(crate) fn transaction_states(states: Vec<SlotState>, selected: &[crate::serve::scheduler::SlotHandle]) {
    if states.len() > 256 {
        invalidate("slot metadata bound exceeded");
        return;
    }
    crate::inference::native_observation::update_owners(&states);
    publish_slot_states(states, "owning_worker_enter_transaction", selected);
}

fn publish_slot_states(states: Vec<SlotState>, cause: &'static str,
    selected: &[crate::serve::scheduler::SlotHandle]) {
    let enabled = match TRACE.get() {
        Some(Trace::Active(lock)) => lock.lock().ok().is_some_and(|r| r.demand_config.is_some()),
        _ => false,
    };
    let slots = if enabled { demand::enrich(&states, cause, selected) }
        else { json!(states) };
    publish("slots", json!({"cause":cause,"slots":slots}));
}

/// One owned worker must finalize the metadata stream. An unhealthy/panicking
/// worker never produces a successful trace footer, even if its process exits.
pub(crate) struct WorkerTrace<F: Fn() -> bool> {
    healthy: F,
    active: bool,
}
pub(crate) fn worker<F: Fn() -> bool>(healthy: F) -> WorkerTrace<F> {
    let active = enabled();
    if let Trace::Active(lock) = TRACE.get_or_init(initialize) {
        if let Ok(mut r) = lock.lock() {
            r.workers += 1;
            if r.workers != 1 {
                r.invalid = Some("ambiguous multiple worker trace owners".into());
            }
        }
    }
    WorkerTrace { healthy, active }
}
impl<F: Fn() -> bool> Drop for WorkerTrace<F> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Trace::Active(lock) = TRACE.get_or_init(initialize) {
            native::final_status();
            if let Ok(mut r) = lock.lock() {
                let error = (std::thread::panicking() || !(self.healthy)())
                    .then_some("worker unhealthy or panicking");
                r.append("end", json!({"error":error,"dropped_events":0}));
                r.sealed = true;
            }
        }
    }
}

pub(crate) fn snapshot() -> Value {
    match TRACE.get_or_init(initialize) {
        Trace::Disabled => json!({"enabled":false}),
        Trace::Invalid(e) => {
            json!({"enabled":true,"schema":SCHEMA,"capture_valid":false,"comparison_qualified":false,"error":e})
        }
        Trace::Active(lock) => match lock.lock() {
            Ok(r) => json!({"enabled":true,"schema":SCHEMA,"config_sha256":r.config_sha256,
                "next_seq":r.sequence,"bytes":r.bytes,"budget_bytes":r.budget,
                "dropped_events":0,"error":r.invalid,"capture_valid":r.invalid.is_none(),
                "sealed":r.sealed,"comparison_qualified":false}),
            Err(_) => {
                json!({"enabled":true,"schema":SCHEMA,"capture_valid":false,"comparison_qualified":false,"error":"trace mutex poisoned"})
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trace_charge_never_trims_or_wraps() {
        assert_eq!(checked_charge(10, 5, 16), Some(16));
        assert_eq!(checked_charge(10, 5, 15), None);
        assert_eq!(checked_charge(u64::MAX, 1, u64::MAX), None);
        assert_eq!(checked_charge(0, EVENT_LIMIT + 1, u64::MAX), None);
    }

    #[test]
    fn invalid_total_quota_refuses_before_any_output_access() {
        let mut c = Config {
            schema: SCHEMA.into(),
            output: PathBuf::from("/must-not-open"),
            run_id: "fixture".into(),
            server_budget_bytes: u64::MAX,
            client_budget_bytes: 128 * 1024 * 1024,
            metadata_budget_profile: None,
            forward_demand_policy: None,
            native_observer: None,
            identity: BTreeMap::new(),
        };
        assert!(validate_config(&c).unwrap_err().contains("quota"));
        c.server_budget_bytes = 129 * 1024 * 1024;
        assert!(validate_config(&c).unwrap_err().contains("quota"));
        c.server_budget_bytes = 128 * 1024 * 1024;
        assert!(validate_config(&c).unwrap_err().contains("identity"));
    }

    #[test]
    fn unknown_configuration_field_cannot_activate_capture() {
        assert!(serde_json::from_value::<Config>(json!({"schema":SCHEMA,
            "output":"/must-not-open","server_budget_bytes":1,"client_budget_bytes":1,
            "identity":{},"silently_drop_events":true}))
        .is_err());
    }

    #[test]
    fn actual_writer_error_is_sticky_and_cannot_emit_a_success_footer() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let number = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("hf2q-f2-trace-{}-{number}", std::process::id()));
        let mut owned = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        owned.write_all(b"unchanged\n").unwrap();
        drop(owned);
        let mut r = Recorder {
            file: File::open(&path).unwrap(),
            origin_ns: monotonic_ns().unwrap(),
            sealed: false,
            workers: 1,
            sequence: 0,
            bytes: 0,
            budget: 4096,
            invalid: None,
            config_sha256: "0".repeat(64),
            native_config: None,
            demand_config: None,
        };
        r.append(
            "progress",
            json!({"fixture":"read-only descriptor, not device work"}),
        );
        let first_error = r.invalid.clone();
        assert!(first_error.is_some());
        r.append("end", json!({"error":null,"dropped_events":0}));
        assert_eq!(r.invalid, first_error);
        assert_eq!(r.sequence, 0);
        assert_eq!(std::fs::read(&path).unwrap(), b"unchanged\n");
        drop(r);
        std::fs::remove_file(path).unwrap();
    }

    fn writable_recorder(tag: &str, budget: u64) -> (PathBuf, Recorder) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let number = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hf2q-f2-trace-{tag}-{}-{number}",
            std::process::id()
        ));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        let recorder = Recorder {
            file,
            origin_ns: monotonic_ns().unwrap(),
            sealed: false,
            workers: 1,
            sequence: 0,
            bytes: 0,
            budget,
            invalid: None,
            config_sha256: "0".repeat(64),
            native_config: None,
            demand_config: None,
        };
        (path, recorder)
    }

    /// Written records with the append clock removed, plus the raw byte count.
    fn written(path: &Path) -> (Vec<Value>, u64) {
        let bytes = std::fs::read(path).unwrap();
        let records = bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| {
                let mut record: Value = serde_json::from_slice(line).unwrap();
                record.as_object_mut().unwrap().remove("at_ns").unwrap();
                record
            })
            .collect();
        (records, bytes.len() as u64)
    }

    #[test]
    fn batched_append_matches_single_appends_and_stops_at_first_failure() {
        let payloads = [
            json!({"command_id":1,"kind":"Submitted"}),
            json!({"command_id":1,"kind":"CheckedDrain"}),
            json!({"command_id":1,"kind":"Dropped","padding":"x".repeat(512)}),
        ];
        for budget in [u64::MAX, 400] {
            let (single_path, mut single) = writable_recorder("single", budget);
            for payload in &payloads {
                single.append("native_event", payload.clone());
            }
            let (batch_path, mut batch) = writable_recorder("batch", budget);
            batch.append_all("native_event", payloads.iter().cloned());
            let (records, length) = written(&batch_path);
            assert_eq!(written(&single_path).0, records);
            assert_eq!(
                (single.sequence, &single.invalid),
                (batch.sequence, &batch.invalid)
            );
            assert_eq!(batch.bytes, length);
            let sequences = records.iter().map(|r| r["seq"].as_u64().unwrap());
            if budget == u64::MAX {
                assert_eq!(sequences.collect::<Vec<_>>(), [0, 1, 2]);
                assert!(batch.invalid.is_none());
            } else {
                // The oversized third record fails its own charge; the first
                // two stay written and the failure is sticky.
                assert_eq!(sequences.collect::<Vec<_>>(), [0, 1]);
                assert!(batch.invalid.as_deref().unwrap().contains("quota"));
                batch.append("end", json!({"error":null,"dropped_events":0}));
                assert_eq!((batch.sequence, written(&batch_path).1), (2, length));
            }
            std::fs::remove_file(single_path).unwrap();
            std::fs::remove_file(batch_path).unwrap();
        }
    }

    #[test]
    fn typed_native_line_matches_json_object_record_bytes() {
        let event: mlx_native::submission_observer::SubmissionEvent =
            serde_json::from_value(json!({
                "clock_id":"CLOCK_MONOTONIC","command_id":51153,"invocation_epoch":44939,
                "queue_identity":4374437424u64,"device_registry_id":4294968845u64,
                "kind":"DispatchEncoded","begin_monotonic_ns":1162576764279000u64,
                "end_monotonic_ns":1162576764279000u64,"dispatch_ordinal":1,
                "noncompute_operations":0,"scope":{"epoch":44939,"id":44965},
                "captured_scope":null,"capture_command_id":null,"capture_generation":null,
                "replay_binding":"Direct","tainted":false,
                "geometry":{"pipeline_identity":532878741888u64,"dispatch_kind":"threads",
                    "grid":[5120,1,1],"threads_per_threadgroup":[256,1,1]},
                "submitted":false,"checked_status":null}))
            .unwrap();
        let line = Line {
            schema: SCHEMA,
            seq: 7,
            at_ns: 9,
            kind: "native_event",
            payload: event,
        };
        assert_eq!(
            serde_json::to_vec(&line).unwrap(),
            serde_json::to_vec(&json!({"schema":SCHEMA,"seq":7,"at_ns":9,
                "kind":"native_event","payload":event}))
            .unwrap()
        );
    }
}
