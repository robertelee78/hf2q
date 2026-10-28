//! ADR-063 F2 bounded metadata observation. Never drives scheduling or admission.
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
}

fn checked_charge(bytes: u64, length: usize, budget: u64) -> Option<u64> {
    if length > EVENT_LIMIT {
        return None;
    }
    bytes
        .checked_add(u64::try_from(length).ok()?.checked_add(1)?)
        .filter(|n| *n <= budget)
}

impl Recorder {
    fn append(&mut self, kind: &str, payload: Value) {
        if self.invalid.is_some() {
            return;
        }
        if self.sealed {
            self.invalid = Some("event after trace finalization".into());
            return;
        }
        let result = (|| -> Result<(), String> {
            let sequence = self.sequence.checked_add(1).ok_or("sequence overflow")?;
            let ns = monotonic_ns()?
                .checked_sub(self.origin_ns)
                .ok_or("monotonic clock regressed")?;
            let line = serde_json::to_vec(&json!({"schema":SCHEMA,"seq":self.sequence,
                "at_ns":ns,"kind":kind,"payload":payload}))
            .map_err(|e| e.to_string())?;
            let bytes = checked_charge(self.bytes, line.len(), self.budget)
                .ok_or("trace byte/event quota exceeded; no trimming or qualification")?;
            self.file
                .write_all(&line)
                .and_then(|_| self.file.write_all(b"\n"))
                .and_then(|_| self.file.flush())
                .map_err(|e| e.to_string())?;
            self.sequence = sequence;
            self.bytes = bytes;
            Ok(())
        })();
        if let Err(error) = result {
            self.invalid = Some(error);
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
        native.validate(c.server_budget_bytes)?;
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

fn monotonic_ns() -> Result<u64, String> {
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
        };
        let mut header = json!({"identity":c.identity,"run_id":c.run_id,
            "server_budget_bytes":c.server_budget_bytes,"client_budget_bytes":c.client_budget_bytes,
            "clock":"CLOCK_MONOTONIC nanoseconds since origin","clock_id":"CLOCK_MONOTONIC","origin_monotonic_ns":recorder.origin_ns,
            "config_sha256":recorder.config_sha256});
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
    publish("slots", json!({"cause":cause,"slots":states}));
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
}
