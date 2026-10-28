//! Source-bound optional native event adapter. No process-wide auto opt-in.
use mlx_native::submission_observer::{ObserverSinkError, SubmissionEvent, SubmissionObserver};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TryRecvError};
use std::sync::Mutex;
use std::sync::{Arc, OnceLock};

const BOOKKEEPING_BYTES: u64 = 1024 * 1024;
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub event_limit: u64,
    pub metadata_limit_bytes: u64,
    /// This source slice supports a separate fresh-process diagnostic, never
    /// transfers its native observations to an observer-OFF primary cohort.
    pub diagnostic_case: bool,
    pub control_nonce: String,
    pub witness_manifest: WitnessManifest,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WitnessManifest {
    pub schema: String,
    pub operator: String,
    pub input_sources: Vec<String>,
    pub coverage: String,
    pub batching_scope: String,
}
impl WitnessManifest {
    fn validate(&self) -> bool {
        let sources = [
            "qwen35.greedy.actual_tokens",
            "qwen35.forward.actual_tokens",
            "gemma4.scalar_forward",
            "gemma4.decode.actual_token_slot_position_arrays",
            "gemma4.prefill.actual_concat_descriptor",
            "gemma4.prefill.actual_single_input",
            "deepseek4.scalar_verifier",
            "deepseek4.prefill.actual_cache_span",
            "deepseek4.sorted_cache_lane",
            "deepseek4.prefill.actual_cooperative_lane",
            "deepseek4.prefill.actual_cooperative_pack",
        ];
        self.schema == "hf2q.f2.forward-witness-policy.v1"
            && self.operator == "target_input_rms_norm"
            && self.coverage == "each_admitted_lane_in_first_fixed_window"
            && self.batching_scope == "operator_only_full_body_unknown"
            && !self.input_sources.is_empty()
            && self.input_sources.len() <= sources.len()
            && self.input_sources.iter().enumerate().all(|(index, value)| {
                sources.contains(&value.as_str()) && !self.input_sources[..index].contains(value)
            })
    }
}
impl Config {
    pub(super) fn reservation_bytes(&self) -> u64 {
        self.metadata_limit_bytes + BOOKKEEPING_BYTES
    }
    pub(super) fn validate(&self, server: u64, event_limit: u64) -> Result<(), String> {
        if !self.diagnostic_case
            || !self.witness_manifest.validate()
            || !(32..=128).contains(&self.control_nonce.len())
            || !self
                .control_nonce
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
            || !(1..=event_limit).contains(&self.event_limit)
            || !(1..=16 * 1024 * 1024).contains(&self.metadata_limit_bytes)
            || self
                .reservation_bytes()
                .checked_add(64 * 1024)
                .filter(|n| *n <= server)
                .is_none()
        {
            return Err("native diagnostic quota must be carved from server budget".into());
        }
        Ok(())
    }
}

const QUEUED_EVENTS: usize = 1024;
static STOP: AtomicBool = AtomicBool::new(false);
static WRITER: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);
struct Sink(SyncSender<SubmissionEvent>);
impl SubmissionObserver for Sink {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        // Fixed-size Copy payload, bounded nonblocking queue; no filesystem,
        // serialization, blocking mutex or native reentry from the callback.
        if STOP.load(Ordering::Acquire) {
            return Err(ObserverSinkError);
        }
        self.0.try_send(*event).map_err(|_| ObserverSinkError)
    }
}

static INSTALL: OnceLock<Result<(), String>> = OnceLock::new();
pub(super) fn install(config: &Config) -> Result<(), String> {
    INSTALL
        .get_or_init(|| {
            if QUEUED_EVENTS * std::mem::size_of::<SubmissionEvent>()
                > BOOKKEEPING_BYTES as usize / 2
            {
                return Err("native event queue exceeds reserved bookkeeping".into());
            }
            let (sender, receiver) = sync_channel(QUEUED_EVENTS);
            let writer = std::thread::Builder::new()
                .name("f2-native-trace".into())
                .stack_size(256 * 1024)
                .spawn(move || loop {
                    match receiver.try_recv() {
                        Ok(event) => super::publish("native_event", serde_json::json!(event)),
                        Err(TryRecvError::Disconnected) => break,
                        Err(TryRecvError::Empty) if STOP.load(Ordering::Acquire) => break,
                        Err(TryRecvError::Empty) => {
                            std::thread::park_timeout(std::time::Duration::from_millis(1))
                        }
                    }
                })
                .map_err(|error| error.to_string())?;
            *WRITER
                .lock()
                .map_err(|_| "native writer ownership poisoned")? = Some(writer);
            mlx_native::submission_observer::install_submission_observer(
                Arc::new(Sink(sender)),
                config.event_limit,
                config.metadata_limit_bytes,
            )
            .map_err(|error| {
                STOP.store(true, Ordering::Release);
                error.to_owned()
            })?;
            crate::inference::native_observation::installed();
            Ok(())
        })
        .clone()
}

pub(super) fn final_status() {
    if INSTALL.get().is_some_and(|result| result.is_ok()) {
        // The owning worker has finished; preserve every queued event before
        // final status/footer. Any late native callback sees STOP and invalidates
        // Core rather than silently disappearing from the finished trace.
        STOP.store(true, Ordering::Release);
        match WRITER.lock() {
            Ok(mut writer) => {
                if writer.take().is_some_and(|worker| worker.join().is_err()) {
                    super::invalidate("native event writer panicked");
                }
            }
            Err(_) => super::invalidate("native writer ownership poisoned"),
        }
        crate::inference::native_observation::finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_budget_is_carved_not_added_to_packet() {
        let config = Config {
            event_limit: 1_000_000,
            metadata_limit_bytes: 16 * 1024 * 1024,
            diagnostic_case: true,
            control_nonce: "n".repeat(32),
            witness_manifest: WitnessManifest {
                schema: "hf2q.f2.forward-witness-policy.v1".into(),
                operator: "target_input_rms_norm".into(),
                input_sources: vec!["qwen35.greedy.actual_tokens".into()],
                coverage: "each_admitted_lane_in_first_fixed_window".into(),
                batching_scope: "operator_only_full_body_unknown".into(),
            },
        };
        assert!(config.validate(128 * 1024 * 1024, 1_000_000).is_ok());
        assert_eq!(
            128 * 1024 * 1024 - config.reservation_bytes(),
            111 * 1024 * 1024
        );
        assert!(config.validate(16 * 1024 * 1024, 1_000_000).is_err());
        assert!(Config {
            diagnostic_case: false,
            ..config
        }
        .validate(128 * 1024 * 1024, 1_000_000)
        .is_err());
    }
    #[test]
    fn required_witness_cannot_be_redeclared_as_auxiliary_or_unbounded_source() {
        let mut manifest = WitnessManifest {
            schema: "hf2q.f2.forward-witness-policy.v1".into(),
            operator: "target_input_rms_norm".into(),
            input_sources: vec!["gemma4.scalar_forward".into()],
            coverage: "each_admitted_lane_in_first_fixed_window".into(),
            batching_scope: "operator_only_full_body_unknown".into(),
        };
        assert!(manifest.validate());
        manifest.operator = "opaque_auxiliary".into();
        assert!(!manifest.validate());
        manifest.operator = "target_input_rms_norm".into();
        manifest
            .input_sources
            .push("unreviewed_arbitrary_source".into());
        assert!(!manifest.validate());
    }
}
