//! hf2q ADR-006 Phase 0 — per-encoder Metal GPU timestamp instrumentation.
//!
//! This module plumbs `MTLCounterSampleBuffer` timestamp sampling into the
//! candle-metal-kernels dispatch path so each compute encoder records its
//! GPU start / end timestamps, keyed by the Metal function name of the
//! pipeline it ran. The aggregated per-kernel table is periodically
//! dumped to a JSON file so the Phase 0 diagnosis script can read it.
//!
//! Design decisions (mantra-aligned, Chesterton-fenced):
//!
//! 1. **Sampling point**: `MTLCounterSamplingPointAtStageBoundary` only.
//!    Probed on M5 Max (applegpu_g17s, 2026-04-11) via a standalone Swift
//!    probe: `supportsCounterSampling(.atStageBoundary)` returns `true`,
//!    `.atDispatchBoundary` returns `false`. This means per-dispatch
//!    sampling via `sampleCountersInBuffer:atSampleIndex:withBarrier:`
//!    on the compute encoder is NOT available on Apple Silicon — we MUST
//!    attach the sample buffer at compute-pass creation time via
//!    `MTLComputePassDescriptor.sampleBufferAttachments[0]` with
//!    `startOfEncoderSampleIndex` / `endOfEncoderSampleIndex`.
//!    Because candle's dispatch pattern is one encoder per kernel call
//!    (verified by tracing `EncoderProvider` call sites in
//!    `src/kernels/quantized.rs` et al.), stage-boundary sampling gives
//!    effectively per-kernel timing despite being a pass-level API.
//!
//! 2. **Sample buffer**: one large `MTLCounterSampleBuffer` with
//!    `SAMPLE_BUFFER_SLOTS = 65536` entries = 32768 encoders worth of
//!    (start, end) pairs. Allocated lazily on first instrumented call
//!    from the system default device. On resolve, we reset `next_slot`
//!    to 0 and reuse the buffer in-place. A single allocation keeps the
//!    observer effect bounded.
//!
//! 3. **Kernel identity**: captured via `MTLComputePipelineState.
//!    computeFunction().name()` at `set_compute_pipeline_state` time.
//!    This is an objc msg send per kernel call — acceptable only under
//!    the instrumented path. Under the uninstrumented path, the
//!    `current_kernel` field is never touched.
//!
//! 4. **Resolution**: pending entries are resolved inside
//!    `Commands::flush_and_wait`. At that call site, all in-flight
//!    command buffers have been `waitUntilCompleted`'d, so every sample
//!    is guaranteed written. After resolution we clear pending and
//!    reset `next_slot` to 0 so the next batch starts fresh.
//!
//! 5. **Dump**: the aggregator is dumped to JSON at the path given by
//!    `HF2Q_PHASE0_INSTRUMENT_DUMP`. Throttled to once every 50ms to
//!    keep I/O overhead negligible. A final dump runs via `atexit`.
//!
//! 6. **Observer-effect gate**: when `HF2Q_PHASE0_INSTRUMENT` is unset
//!    (or equal to `"0"`), `is_enabled()` returns `false` and all new
//!    code paths are short-circuited by a single atomic load. The
//!    uninstrumented path is byte-identical to the pre-patch behavior.
//!    The gate is enforced by the `phase0_candle_bench.sh` script which
//!    runs 5 uninstrumented control runs after the instrumented ones
//!    and fails if median tok/s drifts more than ±2% from 84.9.

use crate::metal::CommandBuffer;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommonCounterSetTimestamp, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLStorageMode,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Number of slots in the single shared counter sample buffer. Each
/// encoder uses 2 slots (start + end), so this supports 2048 encoders
/// between resolutions.
///
/// The upper bound was discovered empirically on M5 Max (2026-04-11):
/// `newCounterSampleBufferWithDescriptor` rejects lengths above 32 KiB
/// with `Invalid sample buffer length: N B. Expected range: 8 -> 32768`.
/// 32 KiB / sizeof(MTLCounterResultTimestamp=8) = 4096, which we stay
/// at the exact ceiling. If a `flush_and_wait` would exceed 2048
/// encoders, the `allocate_slot_pair` path returns `None` and the
/// encoder falls back to the uninstrumented path for that call. The
/// overflow counter is surfaced in the JSON dump so the Phase 0 script
/// can flag under-attribution.
const SAMPLE_BUFFER_SLOTS: usize = 4096;

/// Minimum interval between JSON dumps, in nanoseconds. 50ms is a
/// reasonable balance: enough headroom that the I/O cost is negligible
/// compared to the 10ms/token decode budget, tight enough that a bench
/// script reading the file at the end sees fresh data.
const DUMP_MIN_INTERVAL_NS: u64 = 50_000_000;

/// Unset marker for `start_gpu_time_ns` — set once at the first resolve
/// so we can emit an absolute start timestamp for the bench run.
const UNSET_TIME: u64 = u64::MAX;

/// Fast-path flag. Checked once per `set_compute_pipeline_state` and
/// per dispatch. When `false`, all instrumentation code is bypassed and
/// the vendor crate behaves exactly as it did before the Phase 0 patch.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Has the env-var probe run yet? We do it lazily on the first
/// non-instrumented call rather than at static init because some
/// consumers construct `ComputeCommandEncoder` from tests where no
/// env var is set and we don't want to pay even a `getenv` per call.
static ENABLED_PROBED: AtomicBool = AtomicBool::new(false);

/// Lazily-initialized global instrumentation state. `None` means the
/// env var probe ran and instrumentation is disabled. `Some(_)` means
/// the state is live. Uses `OnceLock` so initialization is race-free.
static STATE: OnceLock<Option<InstrumentState>> = OnceLock::new();

/// Aggregated per-kernel statistics. `total_ns` and `count` are the
/// primary fields; `samples` keeps raw durations so we can compute
/// medians and p95 on dump without needing a separate reservoir.
#[derive(Debug, Default, Clone)]
pub struct KernelStats {
    pub count: u64,
    pub total_ns: u128,
    pub samples: Vec<u64>,
}

impl KernelStats {
    fn record(&mut self, ns: u64) {
        self.count += 1;
        self.total_ns += ns as u128;
        // Cap per-kernel samples at 8192 to keep memory bounded. On a
        // bench with ~12000 calls per kernel per run × 5 runs, 8192 is
        // enough to compute a stable p95.
        if self.samples.len() < 8192 {
            self.samples.push(ns);
        }
    }
}

// SAFETY: Metal objects are thread-safe for the usage patterns in this
// module — we only call `resolveCounterRange:` after `waitUntilCompleted`
// (so no concurrent encode+resolve on the same sample buffer), and the
// device pointer is only used to print its name in the init path. This
// matches the same `unsafe impl Send/Sync` pattern used for `Device`
// and `CommandQueue` elsewhere in this crate.
unsafe impl Send for InstrumentState {}
unsafe impl Sync for InstrumentState {}

struct InstrumentState {
    #[allow(dead_code)] // retained so we outlive the device during sample resolution
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    /// Next free slot in the shared counter sample buffer. Each
    /// encoder allocates 2 slots via `fetch_add(2)` — start and end.
    next_slot: AtomicUsize,
    /// Pending entries awaiting resolution. Each entry is a kernel
    /// name, the start slot index, and the end slot index.
    /// Locked on every dispatch-creation and every flush_and_wait.
    pending: Mutex<Vec<(String, usize, usize)>>,
    /// Resolved per-kernel aggregated statistics.
    aggregator: Mutex<HashMap<String, KernelStats>>,
    /// Path to dump JSON to.
    dump_path: PathBuf,
    /// Monotonic ns clock for throttling dumps.
    dump_epoch: Instant,
    /// Last dump timestamp in ns (relative to `dump_epoch`).
    last_dump_ns: AtomicU64,
    /// First resolved GPU timestamp, for relative t=0 in output.
    start_gpu_time_ns: AtomicU64,
    /// Count of overflow events: encoder started but pending list too
    /// long or next_slot >= SAMPLE_BUFFER_SLOTS. Surfaced in the JSON.
    overflow_count: AtomicU64,
    /// Count of resolves that hit an MTLCounterErrorValue.
    error_value_count: AtomicU64,
}

impl InstrumentState {
    fn new(dump_path: PathBuf) -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;

        // Verify stage-boundary sampling is supported; if not, this
        // device cannot be instrumented at all and we must return
        // None so is_enabled() stays false.
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            eprintln!(
                "[HF2Q_PHASE0_INSTRUMENT] device {} does not support \
                 MTLCounterSamplingPointAtStageBoundary — instrumentation disabled.",
                device.name()
            );
            return None;
        }

        // Locate the timestamp counter set.
        let counter_sets = device.counterSets()?;
        let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        let target_name: &NSString = unsafe { MTLCommonCounterSetTimestamp };
        for i in 0..counter_sets.count() {
            let cs = counter_sets.objectAtIndex(i);
            if cs.name().isEqualToString(target_name) {
                timestamp_set = Some(cs);
                break;
            }
        }
        let timestamp_set = match timestamp_set {
            Some(cs) => cs,
            None => {
                eprintln!(
                    "[HF2Q_PHASE0_INSTRUMENT] device {} exposes no timestamp counter set — instrumentation disabled.",
                    device.name()
                );
                return None;
            }
        };

        // Build the sample buffer descriptor.
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&timestamp_set));
        desc.setStorageMode(MTLStorageMode::Shared);
        unsafe { desc.setSampleCount(SAMPLE_BUFFER_SLOTS) };
        desc.setLabel(&NSString::from_str("hf2q_phase0_instrument"));

        let sample_buffer = match device.newCounterSampleBufferWithDescriptor_error(&desc) {
            Ok(sb) => sb,
            Err(err) => {
                eprintln!(
                    "[HF2Q_PHASE0_INSTRUMENT] newCounterSampleBuffer failed: {} — instrumentation disabled.",
                    err.localizedDescription()
                );
                return None;
            }
        };

        Some(Self {
            device,
            sample_buffer,
            next_slot: AtomicUsize::new(0),
            pending: Mutex::new(Vec::with_capacity(4096)),
            aggregator: Mutex::new(HashMap::new()),
            dump_path,
            dump_epoch: Instant::now(),
            last_dump_ns: AtomicU64::new(0),
            start_gpu_time_ns: AtomicU64::new(UNSET_TIME),
            overflow_count: AtomicU64::new(0),
            error_value_count: AtomicU64::new(0),
        })
    }
}

/// Returns `true` iff Phase 0 instrumentation is active for the current
/// process. Lazily probes `HF2Q_PHASE0_INSTRUMENT` and
/// `HF2Q_PHASE0_INSTRUMENT_DUMP` on first call.
#[inline]
pub fn is_enabled() -> bool {
    if ENABLED_PROBED.load(Ordering::Acquire) {
        return ENABLED.load(Ordering::Relaxed);
    }
    probe_and_init()
}

#[cold]
fn probe_and_init() -> bool {
    // Double-check-lock style: only one thread runs the probe.
    if ENABLED_PROBED.swap(true, Ordering::AcqRel) {
        return ENABLED.load(Ordering::Relaxed);
    }

    let enabled_var = std::env::var("HF2Q_PHASE0_INSTRUMENT").unwrap_or_default();
    if enabled_var.is_empty() || enabled_var == "0" {
        let _ = STATE.set(None);
        ENABLED.store(false, Ordering::Relaxed);
        return false;
    }

    let dump_path = match std::env::var("HF2Q_PHASE0_INSTRUMENT_DUMP") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "[HF2Q_PHASE0_INSTRUMENT] set but HF2Q_PHASE0_INSTRUMENT_DUMP is empty — instrumentation disabled."
            );
            let _ = STATE.set(None);
            ENABLED.store(false, Ordering::Relaxed);
            return false;
        }
    };

    let state = InstrumentState::new(dump_path);
    let live = state.is_some();
    let _ = STATE.set(state);
    ENABLED.store(live, Ordering::Relaxed);

    if live {
        // Register atexit hook for the final dump. Ignoring failure is
        // fine — the throttled runtime dumps already produce a recent
        // snapshot; atexit is just the closing flush.
        register_atexit();
        eprintln!(
            "[HF2Q_PHASE0_INSTRUMENT] active. Dumping to {}",
            STATE.get().unwrap().as_ref().unwrap().dump_path.display()
        );
    }

    live
}

fn state() -> Option<&'static InstrumentState> {
    STATE.get().and_then(|s| s.as_ref())
}

/// hf2q ADR-006 Phase 0 — side-table mapping raw
/// `MTLComputePipelineState` pointer → Metal function name. Populated by
/// `Kernels::load_pipeline_with_constants` at pipeline-cache-miss time
/// (i.e., once per unique kernel) and read by
/// `current_kernel_name` during instrumented encoder dispatch.
///
/// Pointer-keying is safe because `ComputePipeline` is stored in
/// `Kernels::pipelines: RwLock<HashMap>` for the entire process
/// lifetime, so the pointer never dangles.
static PIPELINE_NAMES: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();

fn pipeline_names() -> &'static Mutex<HashMap<usize, String>> {
    PIPELINE_NAMES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register the Metal function name behind a pipeline state pointer.
/// Called unconditionally from `Kernels::load_pipeline_with_constants`
/// — the registration cost is paid once per unique pipeline, not per
/// dispatch, and the table stays tiny (O(kernel_count) ~= 100 entries).
///
/// The registration path intentionally runs even when instrumentation
/// is OFF so that an instrumentation run in the same process would see
/// every pipeline that has been constructed. The cost is one Mutex
/// acquisition per pipeline cache miss — negligible next to the
/// pipeline compilation it follows.
pub fn register_pipeline_name(ptr: usize, name: String) {
    if let Ok(mut t) = pipeline_names().lock() {
        t.insert(ptr, name);
    }
}

/// Called from the sticky compute encoder when a compute pipeline is
/// bound. Looks up the kernel name via `PIPELINE_NAMES`.
/// `MTLComputePipelineState` does not expose its source function and
/// its `label` is read-only (only settable at descriptor creation
/// time), so a raw-pointer side-table is the only kernel-identity
/// channel we have.
pub fn current_kernel_name(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Option<String> {
    if !is_enabled() {
        return None;
    }
    let ptr = pipeline as *const _ as *const () as usize;
    pipeline_names().lock().ok()?.get(&ptr).cloned()
}

/// Allocate a (start_slot, end_slot) pair for a new encoder. Returns
/// `None` if the sample buffer is exhausted; the caller should then
/// treat this encoder as uninstrumented for the current batch. Once
/// the next `flush_and_wait` resolves and resets, subsequent encoders
/// will be instrumented again.
pub fn allocate_slot_pair() -> Option<(usize, usize)> {
    let s = state()?;
    let start = s.next_slot.fetch_add(2, Ordering::Relaxed);
    if start + 2 > SAMPLE_BUFFER_SLOTS {
        s.overflow_count.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    Some((start, start + 1))
}

/// Return the shared sample buffer so the encoder can attach it to the
/// compute pass descriptor.
pub fn sample_buffer() -> Option<&'static ProtocolObject<dyn MTLCounterSampleBuffer>> {
    state().map(|s| &*s.sample_buffer)
}

/// Record a pending sample for a kernel. Called after the encoder is
/// created but before `end_encoding`.
pub fn record_pending(kernel: String, start_slot: usize, end_slot: usize) {
    let Some(s) = state() else {
        return;
    };
    if let Ok(mut p) = s.pending.lock() {
        p.push((kernel, start_slot, end_slot));
    }
}

/// Called from `Commands::flush_and_wait` after all in-flight command
/// buffers have been `waitUntilCompleted`'d. Resolves every pending
/// entry into the aggregator, then resets `next_slot` to 0 and
/// optionally dumps the aggregator to disk.
pub fn resolve_and_maybe_dump() {
    let Some(s) = state() else {
        return;
    };

    // Snapshot pending and clear atomically.
    let entries: Vec<(String, usize, usize)> = {
        let Ok(mut p) = s.pending.lock() else {
            return;
        };
        if p.is_empty() {
            // Still reset next_slot in case other encoders allocated
            // slots but weren't pushed onto pending (shouldn't happen,
            // but keep the two indices coherent).
            s.next_slot.store(0, Ordering::Relaxed);
            return;
        }
        std::mem::take(&mut *p)
    };

    // Compute the resolve range (min..=max of used slots).
    let mut min_slot = usize::MAX;
    let mut max_slot = 0usize;
    for (_, start, end) in &entries {
        if *start < min_slot {
            min_slot = *start;
        }
        if *end + 1 > max_slot {
            max_slot = *end + 1;
        }
    }
    if max_slot <= min_slot {
        s.next_slot.store(0, Ordering::Relaxed);
        return;
    }

    // resolveCounterRange returns an NSData of
    // sizeof(MTLCounterResultTimestamp) per slot = 8 bytes per slot.
    let range = NSRange {
        location: min_slot,
        length: max_slot - min_slot,
    };
    let data = match unsafe { s.sample_buffer.resolveCounterRange(range) } {
        Some(d) => d,
        None => {
            s.error_value_count.fetch_add(1, Ordering::Relaxed);
            s.next_slot.store(0, Ordering::Relaxed);
            return;
        }
    };

    // Copy the NSData into a Vec<u64>.
    let byte_count = data.length();
    let expected_bytes = (max_slot - min_slot) * 8;
    if byte_count != expected_bytes {
        eprintln!(
            "[HF2Q_PHASE0_INSTRUMENT] resolveCounterRange returned {} bytes, expected {}",
            byte_count, expected_bytes
        );
        s.next_slot.store(0, Ordering::Relaxed);
        return;
    }
    let mut timestamps: Vec<u64> = vec![0u64; max_slot - min_slot];
    unsafe {
        let dst = std::ptr::NonNull::new(timestamps.as_mut_ptr() as *mut core::ffi::c_void)
            .expect("timestamp vec ptr");
        data.getBytes_length(dst, byte_count);
    }

    // Record start_gpu_time_ns.
    if s.start_gpu_time_ns.load(Ordering::Relaxed) == UNSET_TIME {
        if let Some(&first) = timestamps.first() {
            if first != MTL_COUNTER_ERROR_VALUE && first != 0 {
                s.start_gpu_time_ns.store(first, Ordering::Relaxed);
            }
        }
    }

    // Resolve each pending entry.
    {
        let Ok(mut agg) = s.aggregator.lock() else {
            return;
        };
        for (kernel, start_slot, end_slot) in entries {
            let start_idx = start_slot - min_slot;
            let end_idx = end_slot - min_slot;
            if end_idx >= timestamps.len() {
                s.error_value_count.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let t_start = timestamps[start_idx];
            let t_end = timestamps[end_idx];
            if t_start == MTL_COUNTER_ERROR_VALUE || t_end == MTL_COUNTER_ERROR_VALUE {
                s.error_value_count.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if t_end < t_start {
                s.error_value_count.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let duration_ns = t_end - t_start;
            let entry = agg.entry(kernel).or_default();
            entry.record(duration_ns);
        }
    }

    // Reset for the next batch.
    s.next_slot.store(0, Ordering::Relaxed);

    // Throttled dump.
    let now = s.dump_epoch.elapsed().as_nanos() as u64;
    let last = s.last_dump_ns.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= DUMP_MIN_INTERVAL_NS {
        if s
            .last_dump_ns
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            dump_snapshot(s);
        }
    }
}

/// Apple's documented "no sample here" sentinel, per
/// MTLCounterErrorValue. Declared inline rather than via a binding
/// because objc2-metal 0.3.2 does not expose the constant.
const MTL_COUNTER_ERROR_VALUE: u64 = u64::MAX;

fn dump_snapshot(s: &InstrumentState) {
    let agg = match s.aggregator.lock() {
        Ok(a) => a.clone(),
        Err(_) => return,
    };

    // Build JSON by hand — avoiding serde_json keeps the vendor crate
    // dep-free for this diagnostic patch. Schema matches
    // docs/phase0-candle-perkernel.json's `kernels` array.
    let mut json = String::with_capacity(4096);
    json.push_str("{\n");
    json.push_str("  \"stack\": \"hf2q+candle\",\n");
    json.push_str(&format!(
        "  \"overflow_count\": {},\n",
        s.overflow_count.load(Ordering::Relaxed)
    ));
    json.push_str(&format!(
        "  \"error_value_count\": {},\n",
        s.error_value_count.load(Ordering::Relaxed)
    ));
    let first_ts = s.start_gpu_time_ns.load(Ordering::Relaxed);
    if first_ts != UNSET_TIME {
        json.push_str(&format!("  \"first_gpu_ns\": {},\n", first_ts));
    }
    json.push_str("  \"kernels\": [\n");
    let mut kernels: Vec<(&String, &KernelStats)> = agg.iter().collect();
    kernels.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
    for (i, (name, stats)) in kernels.iter().enumerate() {
        let mut samples = stats.samples.clone();
        samples.sort_unstable();
        let median = percentile(&samples, 50);
        let p95 = percentile(&samples, 95);
        let p99 = percentile(&samples, 99);
        let max = samples.last().copied().unwrap_or(0);
        let min = samples.first().copied().unwrap_or(0);
        let mean_ns = if stats.count > 0 {
            stats.total_ns as f64 / stats.count as f64
        } else {
            0.0
        };
        if i > 0 {
            json.push_str(",\n");
        }
        json.push_str(&format!(
            "    {{\n      \"name\": \"{}\",\n      \"calls\": {},\n      \"total_ns\": {},\n      \"mean_ns\": {:.1},\n      \"median_ns\": {},\n      \"p95_ns\": {},\n      \"p99_ns\": {},\n      \"min_ns\": {},\n      \"max_ns\": {},\n      \"samples_recorded\": {}\n    }}",
            name, stats.count, stats.total_ns, mean_ns, median, p95, p99, min, max, samples.len()
        ));
    }
    json.push_str("\n  ]\n}\n");

    // Write atomically via temp-then-rename to avoid readers catching a
    // partial file.
    let tmp_path = s.dump_path.with_extension("json.tmp");
    if let Some(parent) = tmp_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match File::create(&tmp_path) {
        Ok(mut f) => {
            if f.write_all(json.as_bytes()).is_ok() {
                let _ = std::fs::rename(&tmp_path, &s.dump_path);
            }
        }
        Err(e) => {
            eprintln!(
                "[HF2Q_PHASE0_INSTRUMENT] failed to open {} for dump: {}",
                tmp_path.display(),
                e
            );
        }
    }
}

fn percentile(sorted: &[u64], pct: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    // Nearest-rank method.
    let rank = ((pct as f64 / 100.0) * (sorted.len() as f64)).ceil() as usize;
    let idx = rank.clamp(1, sorted.len()) - 1;
    sorted[idx]
}

/// Public — force a final resolution and dump. Called from the atexit
/// hook and may be called manually from tests.
pub fn finalize() {
    if !is_enabled() {
        return;
    }
    resolve_and_maybe_dump();
    if let Some(s) = state() {
        // Bypass the throttle for the final dump so the script always
        // sees the latest data.
        s.last_dump_ns.store(0, Ordering::Relaxed);
        dump_snapshot(s);
    }
}

// --- atexit registration -------------------------------------------------
//
// libc::atexit runs registered C functions on normal process exit. We
// use it to guarantee a final dump even if the bench binary doesn't
// explicitly call `finalize()`. The callback signature is
// `extern "C" fn()`; we avoid Rust panics inside it by wrapping the
// body in `catch_unwind`.

extern "C" {
    fn atexit(cb: extern "C" fn()) -> i32;
}

extern "C" fn atexit_hook() {
    let _ = std::panic::catch_unwind(|| {
        finalize();
    });
}

fn register_atexit() {
    unsafe {
        atexit(atexit_hook);
    }
}

// --- encoder creation helper --------------------------------------------
//
// Creates a compute command encoder with a MTLComputePassDescriptor
// carrying the shared sample buffer attached to attachment index 0.
// Returns the raw encoder plus the (start, end) slot indices. The
// caller is responsible for calling `record_pending` after the dispatch
// is encoded and before `end_encoding`.
//
// On any failure (slot exhaustion, descriptor alloc failure, etc.),
// returns `None` and the caller should fall back to the uninstrumented
// path (`MTLCommandBuffer.computeCommandEncoder()`).

pub struct InstrumentedEncoder {
    pub raw: Retained<ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>,
    pub start_slot: usize,
    pub end_slot: usize,
}

pub fn make_instrumented_encoder(cb: &CommandBuffer) -> Option<InstrumentedEncoder> {
    if !is_enabled() {
        return None;
    }
    let s = state()?;

    let (start_slot, end_slot) = allocate_slot_pair()?;

    let desc = MTLComputePassDescriptor::computePassDescriptor();
    let attachments = desc.sampleBufferAttachments();
    let attach = unsafe { attachments.objectAtIndexedSubscript(0) };
    attach.setSampleBuffer(Some(&s.sample_buffer));
    unsafe {
        attach.setStartOfEncoderSampleIndex(start_slot);
        attach.setEndOfEncoderSampleIndex(end_slot);
    }

    let raw_cb: &ProtocolObject<dyn MTLCommandBuffer> = cb.as_ref();
    let raw = raw_cb.computeCommandEncoderWithDescriptor(&desc)?;

    Some(InstrumentedEncoder {
        raw,
        start_slot,
        end_slot,
    })
}
