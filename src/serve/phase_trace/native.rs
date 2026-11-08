//! Source-bound optional native event adapter. No process-wide auto opt-in.
use mlx_native::submission_observer::{
    CompletedGpuTimingEvent, ObserverSinkError, SubmissionEvent, SubmissionObserver,
};
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
    /// Q4-TRACE-GPU-TIMING opt-in: absent (off) or `true`. An explicit `false`
    /// is refused so an off-path config and header stay byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_timestamps: Option<bool>,
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
            "qwen35.joint_decode.actual_slot_rows",
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
    pub(super) fn gpu_timestamps(&self) -> bool {
        self.gpu_timestamps == Some(true)
    }
    pub(super) fn reservation_bytes(&self) -> u64 {
        self.metadata_limit_bytes + BOOKKEEPING_BYTES
    }
    pub(super) fn validate(&self, server: u64, event_limit: u64) -> Result<(), String> {
        if self.gpu_timestamps == Some(false) {
            return Err("native gpu_timestamps is absent or true, never false".into());
        }
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

/// A full queue fails the sink and invalidates the whole capture. Multi-slot
/// decode submits about 1,000 events per 2 ms at peak; this bound absorbs such
/// a burst during a writer park or stall and stays within half the reserved
/// bookkeeping (checked at install).
const QUEUED_EVENTS: usize = 2048;
/// Events per trace lock and file write. One write per event cannot keep up
/// with multi-slot decode submission; a bounded batch amortizes the write
/// without holding the trace lock long against the worker's own records.
const WRITE_BATCH: usize = 64;
static STOP: AtomicBool = AtomicBool::new(false);
/// The installed diagnostic's GPU-timestamp opt-in, cached at install so hot
/// paths never take the trace lock to read it.
static GPU_TIMESTAMPS: AtomicBool = AtomicBool::new(false);
pub(super) fn gpu_timestamps_installed() -> bool {
    GPU_TIMESTAMPS.load(Ordering::Acquire)
}
static WRITER: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);
/// One ordered queue for both native streams: mlx-native emits a command's
/// timing right after its checked drain and before its drop, and a single
/// queue keeps that causal order in the trace.
#[derive(Clone, Copy)]
enum Item {
    Event(SubmissionEvent),
    Timing(CompletedGpuTimingEvent),
}
impl Item {
    fn event(&self) -> Option<&SubmissionEvent> {
        match self {
            Item::Event(event) => Some(event),
            Item::Timing(_) => None,
        }
    }
    fn timing(&self) -> Option<&CompletedGpuTimingEvent> {
        match self {
            Item::Timing(event) => Some(event),
            Item::Event(_) => None,
        }
    }
}
/// Maximal same-kind runs of `items`, in order: `(is_timing, range)`.
fn runs(items: &[Item]) -> impl Iterator<Item = (bool, std::ops::Range<usize>)> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        let first = items.get(start)?;
        let timing = matches!(first, Item::Timing(_));
        let end = items[start..]
            .iter()
            .position(|item| matches!(item, Item::Timing(_)) != timing)
            .map_or(items.len(), |offset| start + offset);
        let run = (timing, start..end);
        start = end;
        Some(run)
    })
}
fn publish_items(items: &[Item]) {
    for (timing, range) in runs(items) {
        if timing {
            super::publish_all(
                "native_gpu_timing",
                items[range].iter().filter_map(Item::timing),
            );
        } else {
            super::publish_all("native_event", items[range].iter().filter_map(Item::event));
        }
    }
}
struct Sink {
    queue: SyncSender<Item>,
    gpu_timestamps: bool,
}
impl Sink {
    fn send(&self, item: Item) -> Result<(), ObserverSinkError> {
        // Fixed-size Copy payload, bounded nonblocking queue; no filesystem,
        // serialization, blocking mutex or native reentry from the callback.
        if STOP.load(Ordering::Acquire) {
            return Err(ObserverSinkError);
        }
        self.queue.try_send(item).map_err(|_| ObserverSinkError)
    }
}
impl SubmissionObserver for Sink {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        self.send(Item::Event(*event))
    }
    fn record_gpu_timing(&self, event: &CompletedGpuTimingEvent) -> Result<(), ObserverSinkError> {
        if !self.gpu_timestamps {
            return Err(ObserverSinkError);
        }
        self.send(Item::Timing(*event))
    }
}

static INSTALL: OnceLock<Result<(), String>> = OnceLock::new();
pub(super) fn install(config: &Config) -> Result<(), String> {
    INSTALL
        .get_or_init(|| {
            if QUEUED_EVENTS * std::mem::size_of::<Item>() > BOOKKEEPING_BYTES as usize / 2 {
                return Err("native event queue exceeds reserved bookkeeping".into());
            }
            let (sender, receiver) = sync_channel(QUEUED_EVENTS);
            let writer = std::thread::Builder::new()
                .name("f2-native-trace".into())
                .stack_size(256 * 1024)
                .spawn(move || {
                    let mut batch = Vec::with_capacity(WRITE_BATCH);
                    loop {
                        let mut idle = None;
                        while batch.len() < WRITE_BATCH {
                            match receiver.try_recv() {
                                Ok(event) => batch.push(event),
                                Err(error) => {
                                    idle = Some(error);
                                    break;
                                }
                            }
                        }
                        if !batch.is_empty() {
                            publish_items(&batch);
                            batch.clear();
                            continue;
                        }
                        match idle {
                            Some(TryRecvError::Disconnected) => break,
                            Some(TryRecvError::Empty) if STOP.load(Ordering::Acquire) => break,
                            _ => std::thread::park_timeout(std::time::Duration::from_millis(1)),
                        }
                    }
                })
                .map_err(|error| error.to_string())?;
            *WRITER
                .lock()
                .map_err(|_| "native writer ownership poisoned")? = Some(writer);
            let sink = Arc::new(Sink {
                queue: sender,
                gpu_timestamps: config.gpu_timestamps(),
            });
            let installed = if config.gpu_timestamps() {
                mlx_native::submission_observer::install_submission_observer_with_gpu_timestamps(
                    sink,
                    config.event_limit,
                    config.metadata_limit_bytes,
                )
            } else {
                mlx_native::submission_observer::install_submission_observer(
                    sink,
                    config.event_limit,
                    config.metadata_limit_bytes,
                )
            };
            installed.map_err(|error| {
                STOP.store(true, Ordering::Release);
                error.to_owned()
            })?;
            GPU_TIMESTAMPS.store(config.gpu_timestamps(), Ordering::Release);
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
    fn queue_and_write_batch_fit_reserved_bookkeeping() {
        // `install` refuses a queue past half the bookkeeping; a native pin
        // that grows the event must fail here, not at a traced launch.
        let item = std::mem::size_of::<Item>();
        assert!(
            item <= std::mem::size_of::<SubmissionEvent>()
                .max(std::mem::size_of::<CompletedGpuTimingEvent>())
                + 8
        );
        assert!(QUEUED_EVENTS * item <= BOOKKEEPING_BYTES as usize / 2);
        assert!((QUEUED_EVENTS + WRITE_BATCH) * item <= BOOKKEEPING_BYTES as usize);
    }
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
            gpu_timestamps: None,
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
    #[test]
    fn qwen_traced_manifest_admits_the_joint_decode_source_once() {
        let mut manifest = WitnessManifest {
            schema: "hf2q.f2.forward-witness-policy.v1".into(),
            operator: "target_input_rms_norm".into(),
            input_sources: vec![
                "qwen35.greedy.actual_tokens".into(),
                "qwen35.forward.actual_tokens".into(),
                "qwen35.joint_decode.actual_slot_rows".into(),
            ],
            coverage: "each_admitted_lane_in_first_fixed_window".into(),
            batching_scope: "operator_only_full_body_unknown".into(),
        };
        assert!(manifest.validate());
        manifest
            .input_sources
            .push("qwen35.joint_decode.actual_slot_rows".into());
        assert!(!manifest.validate());
    }
    fn sample_config(gpu_timestamps: Option<bool>) -> Config {
        Config {
            event_limit: 1_000_000,
            metadata_limit_bytes: 1024 * 1024,
            diagnostic_case: true,
            control_nonce: "n".repeat(32),
            witness_manifest: WitnessManifest {
                schema: "hf2q.f2.forward-witness-policy.v1".into(),
                operator: "target_input_rms_norm".into(),
                input_sources: vec!["qwen35.greedy.actual_tokens".into()],
                coverage: "each_admitted_lane_in_first_fixed_window".into(),
                batching_scope: "operator_only_full_body_unknown".into(),
            },
            gpu_timestamps,
        }
    }
    #[test]
    fn gpu_timestamps_is_absent_or_true_and_off_configs_serialize_unchanged() {
        let off = sample_config(None);
        let json = serde_json::to_value(&off).unwrap();
        assert!(json.get("gpu_timestamps").is_none());
        let parsed: Config = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), json);
        assert!(!off.gpu_timestamps());
        let on = sample_config(Some(true));
        assert!(on.gpu_timestamps());
        assert_eq!(serde_json::to_value(&on).unwrap()["gpu_timestamps"], true);
        assert!(on.validate(128 * 1024 * 1024, 1_000_000).is_ok());
        assert!(sample_config(Some(false))
            .validate(128 * 1024 * 1024, 1_000_000)
            .is_err());
        let not_diagnostic = Config {
            diagnostic_case: false,
            ..sample_config(Some(true))
        };
        assert!(not_diagnostic
            .validate(128 * 1024 * 1024, 1_000_000)
            .is_err());
    }
    fn event(command: u64) -> SubmissionEvent {
        serde_json::from_value(serde_json::json!({
            "clock_id":"CLOCK_MONOTONIC","command_id":command,"invocation_epoch":3,
            "queue_identity":4,"device_registry_id":5,"kind":"CheckedDrain",
            "begin_monotonic_ns":1,"end_monotonic_ns":2,"dispatch_ordinal":1,
            "noncompute_operations":0,"scope":null,"captured_scope":null,
            "capture_command_id":null,"capture_generation":null,"replay_binding":"Direct",
            "tainted":false,"geometry":null,"submitted":true,"checked_status":4}))
        .unwrap()
    }
    fn timing(command: u64) -> CompletedGpuTimingEvent {
        serde_json::from_value(serde_json::json!({
            "schema_version":1,"command_id":command,"invocation_epoch":3,"queue_identity":4,
            "device_registry_id":5,"timestamps":{"Available":{
                "clock":"MetalCommandBufferHostSecondsSystemMachTime",
                "start_seconds":1.5,"end_seconds":1.75}}}))
        .unwrap()
    }
    #[test]
    fn one_queue_keeps_causal_order_as_same_kind_runs() {
        let items = [
            Item::Event(event(1)),
            Item::Timing(timing(1)),
            Item::Event(event(1)),
            Item::Event(event(2)),
            Item::Timing(timing(2)),
        ];
        let runs: Vec<_> = runs(&items).collect();
        assert_eq!(
            runs,
            [(false, 0..1), (true, 1..2), (false, 2..4), (true, 4..5)]
        );
        assert_eq!(super::runs(&[]).count(), 0);
        // Every item is published exactly once, in order.
        let covered: Vec<_> = runs.iter().flat_map(|(_, range)| range.clone()).collect();
        assert_eq!(covered, (0..items.len()).collect::<Vec<_>>());
    }
    #[test]
    fn timing_sink_refuses_when_off_or_full() {
        let (sender, receiver) = sync_channel(1);
        let off = Sink {
            queue: sender.clone(),
            gpu_timestamps: false,
        };
        assert!(off.record_gpu_timing(&timing(1)).is_err());
        let on = Sink {
            queue: sender,
            gpu_timestamps: true,
        };
        assert!(on.record_gpu_timing(&timing(1)).is_ok());
        assert!(on.record(&event(1)).is_err(), "full queue");
        assert!(matches!(receiver.try_recv(), Ok(Item::Timing(_))));
        assert!(on.record(&event(2)).is_ok());
        assert!(matches!(receiver.try_recv(), Ok(Item::Event(_))));
    }
}
