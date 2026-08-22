use std::cmp::Ordering;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mlx_native::{
    barrier_count, cmd_buf_count, dispatch_count, reset_counters, sync_count, MlxBuffer,
};

use super::cache::{Deepseek4Cache, Deepseek4CacheSnapshot};
use super::model::Deepseek4Model;
use super::real_artifact_tests::official_artifact;

const LANES: usize = 4;
const PREFIX_ROWS: usize = 148;
const DECODE_STEPS: usize = 132;
const PARITY_CAPACITY: usize = PREFIX_ROWS + DECODE_STEPS;
const BENCHMARK_POSITION: usize = 6_676;
const BENCHMARK_LOGICAL_CAPACITY: usize = 131_072;
// Gives the hardware runner enough room to require 30 continuously nominal
// seconds after the model and production-capacity caches are resident.
const LOADED_IDLE_SECONDS: u64 = 45;
const BENCHMARK_PAIRS: usize = 20;
const RECEIPT_ENV: &str = "HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT";
const PHASE_DIR_ENV: &str = "HF2Q_DEEPSEEK4_DECODE_COHORT_PHASE_DIR";
const RUN_UUID_ENV: &str = "HF2Q_DEEPSEEK4_DECODE_COHORT_RUN_UUID";
const PHASE_ACK_FILE: &str = "measurement-armed.ack";
const PHASE_ACK_TIMEOUT_SECONDS: u64 = 300;
const PHYSICAL_TO_LOGICAL: [usize; LANES] = [2, 0, 3, 1];
const BENCHMARK_OVERRIDE_ENV: &[&str] = &[
    "HF2Q_DEEPSEEK_COMPRESSED_STAGE_PROFILE",
    "HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR",
    "HF2Q_DEEPSEEK_DUMP_LAYER_DIR",
    "HF2Q_DEEPSEEK_ENCODER_STAGES",
    "HF2Q_DEEPSEEK_GRAPH_DIAG",
    "HF2Q_DEEPSEEK_LAYERS_PER_CB",
    "HF2Q_DEEPSEEK_MMAP_WEIGHTS",
    "HF2Q_DEEPSEEK_STAGE_PROFILE",
    "HF2Q_MM_ID_ROUTING_THRESHOLD",
    "MLX_PROFILE_CB",
    "MLX_PROFILE_DISPATCH",
    "MLX_UNRETAINED_REFS",
];

#[derive(Clone, Debug)]
struct PhaseMarker {
    run_uuid: String,
    sequence: u64,
    phase: &'static str,
    pid: u32,
    monotonic_ns: u64,
    wall_ns: u64,
}

impl PhaseMarker {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "run_uuid": self.run_uuid,
            "sequence": self.sequence,
            "phase": self.phase,
            "pid": self.pid,
            "monotonic_ns": self.monotonic_ns,
            "wall_ns": self.wall_ns,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct DarwinVmSnapshot {
    boot_time_seconds: i64,
    page_size: u64,
    pressure_level: i32,
    pageins: u64,
    pageouts: u64,
    swapins: u64,
    swapouts: u64,
    compressions: u64,
    decompressions: u64,
    purges: u64,
    reactivations: u64,
    throttled_pages: u64,
    wired_pages: u64,
    compressor_pages: u64,
    uncompressed_compressor_pages: u64,
    process_pageins: u64,
}

impl DarwinVmSnapshot {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "boot_time_seconds": self.boot_time_seconds,
            "page_size": self.page_size,
            "pressure_level": self.pressure_level,
            "pageins": self.pageins,
            "pageouts": self.pageouts,
            "swapins": self.swapins,
            "swapouts": self.swapouts,
            "compressions": self.compressions,
            "decompressions": self.decompressions,
            "purges": self.purges,
            "reactivations": self.reactivations,
            "throttled_pages": self.throttled_pages,
            "wired_pages": self.wired_pages,
            "compressor_pages": self.compressor_pages,
            "uncompressed_compressor_pages": self.uncompressed_compressor_pages,
            "process_pageins": self.process_pageins,
        })
    }
}

fn checked_elapsed_ns(test_started: Instant) -> u64 {
    u64::try_from(test_started.elapsed().as_nanos()).expect("phase monotonic time must fit u64")
}

fn wall_time_ns() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos(),
    )
    .expect("wall-clock nanoseconds must fit u64")
}

fn is_lowercase_uuid_shape(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            }
        })
}

fn write_phase_marker(
    phase_dir: &Path,
    run_uuid: &str,
    sequence: u64,
    phase: &'static str,
    test_started: Instant,
) -> PhaseMarker {
    let marker = PhaseMarker {
        run_uuid: run_uuid.to_owned(),
        sequence,
        phase,
        pid: std::process::id(),
        monotonic_ns: checked_elapsed_ns(test_started),
        wall_ns: wall_time_ns(),
    };
    let name = format!("{sequence:03}-{phase}.json");
    let published = phase_dir.join(&name);
    let temporary = phase_dir.join(format!(".{name}.tmp"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .unwrap_or_else(|error| panic!("create phase marker {}: {error}", temporary.display()));
    let mut encoded = serde_json::to_vec(&marker.json()).expect("serialize phase marker");
    encoded.push(b'\n');
    file.write_all(&encoded)
        .unwrap_or_else(|error| panic!("write phase marker {}: {error}", temporary.display()));
    file.sync_all()
        .unwrap_or_else(|error| panic!("fsync phase marker {}: {error}", temporary.display()));
    drop(file);
    std::fs::rename(&temporary, &published).unwrap_or_else(|error| {
        panic!(
            "publish phase marker {} -> {}: {error}",
            temporary.display(),
            published.display()
        )
    });
    std::fs::File::open(phase_dir)
        .and_then(|directory| directory.sync_all())
        .unwrap_or_else(|error| panic!("fsync phase directory {}: {error}", phase_dir.display()));
    eprintln!(
        "HF2Q_DEEPSEEK4_PHASE {}",
        serde_json::to_string(&marker.json()).unwrap()
    );
    marker
}

fn wait_for_measurement_ack(phase_dir: &Path, run_uuid: &str) {
    let ack = phase_dir.join(PHASE_ACK_FILE);
    let deadline = Instant::now() + Duration::from_secs(PHASE_ACK_TIMEOUT_SECONDS);
    loop {
        match std::fs::read_to_string(&ack) {
            Ok(value) => {
                assert_eq!(
                    value.trim(),
                    run_uuid,
                    "measurement acknowledgement is bound to the wrong run"
                );
                return;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!(
                "read measurement acknowledgement {}: {error}",
                ack.display()
            ),
        }
        assert!(
            Instant::now() < deadline,
            "runner did not acknowledge measurement readiness within {PHASE_ACK_TIMEOUT_SECONDS}s"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc exposes the system Mach call; this is test-only telemetry.
fn capture_darwin_vm_snapshot() -> DarwinVmSnapshot {
    use std::ffi::CString;
    use std::mem::{size_of, zeroed};

    unsafe fn sysctl_value<T: Copy>(name: &str) -> T {
        let name = CString::new(name).expect("sysctl name must not contain NUL");
        let mut value = std::mem::MaybeUninit::<T>::uninit();
        let mut size = size_of::<T>();
        let rc = libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        assert_eq!(rc, 0, "sysctlbyname failed for {}", name.to_string_lossy());
        assert_eq!(
            size,
            size_of::<T>(),
            "sysctl size drift for {}",
            name.to_string_lossy()
        );
        value.assume_init()
    }

    let pressure_level = unsafe { sysctl_value::<i32>("kern.memorystatus_vm_pressure_level") };
    let boot_time = unsafe { sysctl_value::<libc::timeval>("kern.boottime") };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(page_size > 0, "sysconf(_SC_PAGESIZE) failed");

    let mut vm = unsafe { zeroed::<libc::vm_statistics64>() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let host_rc = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            (&mut vm as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    assert_eq!(host_rc, libc::KERN_SUCCESS, "host_statistics64 failed");
    assert_eq!(
        count,
        libc::HOST_VM_INFO64_COUNT,
        "HOST_VM_INFO64 count drift"
    );

    let mut process = unsafe { zeroed::<libc::rusage_info_v4>() };
    let process_rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut process as *mut libc::rusage_info_v4).cast(),
        )
    };
    assert_eq!(process_rc, 0, "proc_pid_rusage(RUSAGE_INFO_V4) failed");

    DarwinVmSnapshot {
        boot_time_seconds: boot_time.tv_sec,
        page_size: u64::try_from(page_size).expect("page size must fit u64"),
        pressure_level,
        pageins: vm.pageins,
        pageouts: vm.pageouts,
        swapins: vm.swapins,
        swapouts: vm.swapouts,
        compressions: vm.compressions,
        decompressions: vm.decompressions,
        purges: vm.purges,
        reactivations: vm.reactivations,
        throttled_pages: u64::from(vm.throttled_count),
        wired_pages: u64::from(vm.wire_count),
        compressor_pages: u64::from(vm.compressor_page_count),
        uncompressed_compressor_pages: vm.total_uncompressed_pages_in_compressor,
        process_pageins: process.ri_pageins,
    }
}

#[cfg(not(target_os = "macos"))]
fn capture_darwin_vm_snapshot() -> DarwinVmSnapshot {
    panic!("protected DeepSeek-V4 decode measurement requires macOS")
}

#[derive(Clone, Copy, Debug)]
struct TrialCounters {
    command_buffers: u64,
    synchronizations: u64,
    dispatches: u64,
    barriers: u64,
}

fn prefix_tokens(logical_lane: usize) -> Vec<u32> {
    (0..PREFIX_ROWS)
        .map(|row| ((row * 97 + logical_lane * 193 + 11) % 120_000) as u32)
        .collect()
}

fn supplied_tokens(step: usize) -> [u32; LANES] {
    std::array::from_fn(|logical_lane| ((step * 389 + logical_lane * 7_919 + 17) % 120_000) as u32)
}

fn benchmark_extension(logical_lane: usize) -> Vec<u32> {
    (PARITY_CAPACITY..BENCHMARK_POSITION)
        .map(|position| ((position * 521 + logical_lane * 7_919 + 29) % 120_000) as u32)
        .collect()
}

fn grow_and_extend_caches(
    model: &mut Deepseek4Model,
    caches: Vec<Deepseek4Cache>,
    physical_to_logical: [usize; LANES],
) -> Vec<Deepseek4Cache> {
    caches
        .into_iter()
        .enumerate()
        .map(|(physical_lane, source)| {
            let logical_lane = physical_to_logical[physical_lane];
            let mut grown = model
                .allocate_logical_cache(BENCHMARK_LOGICAL_CAPACITY)
                .expect("allocate production-capacity benchmark cache");
            grown
                .migrate_from(&source, None)
                .expect("grow parity cache into production logical capacity");
            model
                .forward_verifier_prompt(&benchmark_extension(logical_lane), &mut grown)
                .expect("extend benchmark cache to production anchor");
            assert_eq!(grown.position(), BENCHMARK_POSITION);
            assert_eq!(grown.capacity(), BENCHMARK_LOGICAL_CAPACITY);
            grown
        })
        .collect()
}

fn read_f32(buffer: &MlxBuffer, label: &str) -> Vec<f32> {
    buffer
        .as_logical_slice::<f32>()
        .unwrap_or_else(|error| panic!("read {label}: {error}"))
        .to_vec()
}

fn assert_exact_f32(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label} length differs");
    let mismatch = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
        .map(|(index, (expected, actual))| (index, *expected, *actual));
    assert!(mismatch.is_none(), "{label} differs: {mismatch:?}");
}

fn valid_row_bytes(buffer: &MlxBuffer, rows: usize, label: &str) -> usize {
    let row_elements = buffer.shape()[1..]
        .iter()
        .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension))
        .unwrap_or_else(|| panic!("{label} row size overflow"));
    rows.checked_mul(row_elements)
        .and_then(|elements| elements.checked_mul(buffer.dtype().size_of()))
        .unwrap_or_else(|| panic!("{label} byte size overflow"))
}

fn assert_buffer_prefix_exact(
    label: &str,
    expected: &MlxBuffer,
    actual: &MlxBuffer,
    valid_bytes: usize,
) {
    assert_eq!(expected.dtype(), actual.dtype(), "{label} dtype differs");
    assert_eq!(expected.shape(), actual.shape(), "{label} shape differs");
    assert!(
        valid_bytes <= expected.data_byte_len() && valid_bytes <= actual.data_byte_len(),
        "{label} valid byte bound exceeds allocation"
    );
    let expected = &expected
        .as_logical_slice::<u8>()
        .unwrap_or_else(|error| panic!("read expected {label}: {error}"))[..valid_bytes];
    let actual = &actual
        .as_logical_slice::<u8>()
        .unwrap_or_else(|error| panic!("read actual {label}: {error}"))[..valid_bytes];
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual);
    assert!(mismatch.is_none(), "{label} differs at byte {mismatch:?}");
}

fn assert_optional_buffer_exact(
    label: &str,
    expected: Option<&MlxBuffer>,
    actual: Option<&MlxBuffer>,
    valid_bytes: usize,
) {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            assert_buffer_prefix_exact(label, expected, actual, valid_bytes)
        }
        (None, None) => assert_eq!(valid_bytes, 0, "{label} unexpectedly has valid bytes"),
        _ => panic!("{label} optional-buffer presence differs"),
    }
}

fn assert_cache_exact(label: &str, expected: &Deepseek4Cache, actual: &Deepseek4Cache) {
    assert_eq!(
        expected.position(),
        actual.position(),
        "{label} cursor differs"
    );
    assert_eq!(
        expected.capacity(),
        actual.capacity(),
        "{label} capacity differs"
    );
    assert_eq!(expected.plan, actual.plan, "{label} plan differs");
    assert_eq!(
        expected.is_poisoned(),
        actual.is_poisoned(),
        "{label} poison state differs"
    );

    let position = expected.position();
    for (layer_index, ((layer_plan, expected), actual)) in expected
        .plan
        .layers
        .iter()
        .zip(expected.layers())
        .zip(actual.layers())
        .enumerate()
    {
        let window_rows = position.min(expected.window_kv.shape()[0]);
        assert_buffer_prefix_exact(
            &format!("{label} layer {layer_index} window"),
            &expected.window_kv,
            &actual.window_kv,
            valid_row_bytes(&expected.window_kv, window_rows, "window"),
        );
        let compressed_rows = if layer_plan.compress_ratio == 0 {
            0
        } else {
            position / layer_plan.compress_ratio as usize
        };
        let compressed_bytes = expected.compressed_kv.as_ref().map_or(0, |buffer| {
            valid_row_bytes(buffer, compressed_rows, "compressed KV")
        });
        assert_optional_buffer_exact(
            &format!("{label} layer {layer_index} compressed"),
            expected.compressed_kv.as_ref(),
            actual.compressed_kv.as_ref(),
            compressed_bytes,
        );
        let indexer_rows = if layer_plan.compress_ratio == 4 {
            compressed_rows
        } else {
            0
        };
        let indexer_bytes = expected.indexer_kv.as_ref().map_or(0, |buffer| {
            valid_row_bytes(buffer, indexer_rows, "indexer KV")
        });
        assert_optional_buffer_exact(
            &format!("{label} layer {layer_index} indexer"),
            expected.indexer_kv.as_ref(),
            actual.indexer_kv.as_ref(),
            indexer_bytes,
        );
        for (kind, expected, actual) in [
            (
                "main KV state",
                expected.main_kv_state.as_ref(),
                actual.main_kv_state.as_ref(),
            ),
            (
                "main score state",
                expected.main_score_state.as_ref(),
                actual.main_score_state.as_ref(),
            ),
            (
                "indexer KV state",
                expected.indexer_kv_state.as_ref(),
                actual.indexer_kv_state.as_ref(),
            ),
            (
                "indexer score state",
                expected.indexer_score_state.as_ref(),
                actual.indexer_score_state.as_ref(),
            ),
        ] {
            let valid_bytes = expected.map_or(0, MlxBuffer::data_byte_len);
            assert_optional_buffer_exact(
                &format!("{label} layer {layer_index} {kind}"),
                expected,
                actual,
                valid_bytes,
            );
        }
    }
}

fn assert_decode_step_exact(
    label: &str,
    model: &mut Deepseek4Model,
    serial_caches: &mut [Deepseek4Cache],
    cohort_caches: &mut [Deepseek4Cache],
    logical_tokens: [u32; LANES],
) {
    let mut serial_states = Vec::with_capacity(LANES);
    let mut serial_logits = Vec::with_capacity(LANES);
    for logical_lane in 0..LANES {
        let state = model
            .forward_verifier_one(
                logical_tokens[logical_lane],
                &mut serial_caches[logical_lane],
            )
            .unwrap_or_else(|error| panic!("{label} serial lane {logical_lane} body: {error:#}"));
        let logits = model
            .forward_logits(&state)
            .unwrap_or_else(|error| panic!("{label} serial lane {logical_lane} head: {error:#}"));
        serial_states.push(read_f32(&state, &format!("{label} serial decode state")));
        serial_logits.push(read_f32(&logits, &format!("{label} serial logits")));
    }

    let physical_tokens = PHYSICAL_TO_LOGICAL.map(|logical_lane| logical_tokens[logical_lane]);
    let [lane0, lane1, lane2, lane3] = cohort_caches else {
        panic!("{label} B=4 cache count drift")
    };
    let mut cache_refs = [lane0, lane1, lane2, lane3];
    let cohort_state = model
        .forward_verifier_decode_cohort(physical_tokens, &mut cache_refs)
        .unwrap_or_else(|error| panic!("{label} B=4 body: {error:#}"));
    let cohort_logits = model
        .forward_logits(&cohort_state)
        .unwrap_or_else(|error| panic!("{label} B=4 head: {error:#}"));
    assert_eq!(cohort_state.shape(), [LANES, 4, 4_096]);
    assert_eq!(
        cohort_logits.shape(),
        [LANES, model.cfg.vocab_size as usize]
    );
    let cohort_state = read_f32(&cohort_state, &format!("{label} B=4 decode state"));
    let cohort_logits = read_f32(&cohort_logits, &format!("{label} B=4 logits"));
    let state_row = 4 * 4_096;
    let logit_row = model.cfg.vocab_size as usize;
    for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
        assert_exact_f32(
            &format!("{label} logical lane {logical_lane} state"),
            &serial_states[logical_lane],
            &cohort_state[physical_lane * state_row..(physical_lane + 1) * state_row],
        );
        assert_exact_f32(
            &format!("{label} logical lane {logical_lane} logits"),
            &serial_logits[logical_lane],
            &cohort_logits[physical_lane * logit_row..(physical_lane + 1) * logit_row],
        );
        assert_cache_exact(
            &format!("{label} logical lane {logical_lane} cache"),
            &serial_caches[logical_lane],
            &cohort_caches[physical_lane],
        );
    }
}

fn counter_snapshot() -> TrialCounters {
    TrialCounters {
        command_buffers: cmd_buf_count(),
        synchronizations: sync_count(),
        dispatches: dispatch_count(),
        barriers: barrier_count(),
    }
}

fn timed_serial(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    tokens: [u32; LANES],
) -> (f64, TrialCounters) {
    reset_counters();
    let started = Instant::now();
    for logical_lane in 0..LANES {
        let state = model
            .forward_verifier_one(tokens[logical_lane], &mut caches[logical_lane])
            .expect("timed serial decode body");
        let logits = model
            .forward_logits(&state)
            .expect("timed serial output head");
        drop(logits);
        drop(state);
    }
    (
        started.elapsed().as_secs_f64() * 1_000.0,
        counter_snapshot(),
    )
}

fn timed_cohort(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    logical_tokens: [u32; LANES],
) -> (f64, TrialCounters) {
    let physical_tokens = PHYSICAL_TO_LOGICAL.map(|logical_lane| logical_tokens[logical_lane]);
    let [lane0, lane1, lane2, lane3] = caches else {
        panic!("B=4 benchmark cache count drift")
    };
    let mut cache_refs = [lane0, lane1, lane2, lane3];
    reset_counters();
    let started = Instant::now();
    let state = model
        .forward_verifier_decode_cohort(physical_tokens, &mut cache_refs)
        .expect("timed B=4 decode body");
    let logits = model.forward_logits(&state).expect("timed B=4 output head");
    drop(logits);
    drop(state);
    (
        started.elapsed().as_secs_f64() * 1_000.0,
        counter_snapshot(),
    )
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

fn order_stratum(values: &[f64], parity: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (index % 2 == parity).then_some(value))
        .collect()
}

fn paired_deltas(serial_ms: &[f64], cohort_ms: &[f64]) -> Vec<f64> {
    assert_eq!(serial_ms.len(), cohort_ms.len());
    serial_ms
        .iter()
        .zip(cohort_ms)
        .map(|(&serial, &cohort)| serial - cohort)
        .collect()
}

fn restore_caches(
    caches: &mut [Deepseek4Cache],
    snapshots: &[Deepseek4CacheSnapshot],
    label: &str,
) {
    for (cache, snapshot) in caches.iter_mut().zip(snapshots) {
        cache
            .restore(snapshot)
            .unwrap_or_else(|error| panic!("restore {label} cache: {error:#}"));
    }
}

fn timed_serial_conditioned(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    snapshots: &[Deepseek4CacheSnapshot],
    tokens: [u32; LANES],
) -> (f64, TrialCounters, f64, TrialCounters) {
    restore_caches(caches, snapshots, "conditioned serial prime");
    let (prime_elapsed, prime_counters) = timed_serial(model, caches, tokens);
    restore_caches(caches, snapshots, "conditioned serial timing");
    let (elapsed, measured_counters) = timed_serial(model, caches, tokens);
    (prime_elapsed, prime_counters, elapsed, measured_counters)
}

fn timed_cohort_conditioned(
    model: &mut Deepseek4Model,
    caches: &mut [Deepseek4Cache],
    snapshots: &[Deepseek4CacheSnapshot],
    tokens: [u32; LANES],
) -> (f64, TrialCounters, f64, TrialCounters) {
    restore_caches(caches, snapshots, "conditioned cohort prime");
    let (prime_elapsed, prime_counters) = timed_cohort(model, caches, tokens);
    restore_caches(caches, snapshots, "conditioned cohort timing");
    let (elapsed, measured_counters) = timed_cohort(model, caches, tokens);
    (prime_elapsed, prime_counters, elapsed, measured_counters)
}

fn record_counter_errors(
    errors: &mut Vec<String>,
    label: &str,
    counters: TrialCounters,
    expected_command_buffers: u64,
    expected_synchronizations: u64,
) {
    if counters.command_buffers != expected_command_buffers {
        errors.push(format!(
            "{label}: command_buffers={} expected={expected_command_buffers}",
            counters.command_buffers
        ));
    }
    if counters.synchronizations != expected_synchronizations {
        errors.push(format!(
            "{label}: synchronizations={} expected={expected_synchronizations}",
            counters.synchronizations
        ));
    }
    if counters.dispatches == 0 {
        errors.push(format!("{label}: dispatches=0"));
    }
    if counters.barriers == 0 {
        errors.push(format!("{label}: barriers=0"));
    }
}

fn checked_resident_sum(label: &str, values: impl IntoIterator<Item = u64>) -> u64 {
    values
        .into_iter()
        .try_fold(0_u64, |total, value| total.checked_add(value))
        .unwrap_or_else(|| panic!("{label} resident-byte accounting overflow"))
}

#[test]
#[ignore = "loads the official checkpoint and proves the production B=4 decode transaction"]
fn official_artifact_b4_decode_body_is_exact_and_measured() {
    let test_started = Instant::now();
    let receipt_path = std::env::var_os(RECEIPT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("protected B=4 decode proof requires {RECEIPT_ENV}"));
    let phase_dir = std::env::var_os(PHASE_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("protected B=4 decode proof requires {PHASE_DIR_ENV}"));
    let run_uuid = std::env::var(RUN_UUID_ENV)
        .unwrap_or_else(|_| panic!("protected B=4 decode proof requires {RUN_UUID_ENV}"));
    assert!(phase_dir.is_dir(), "phase directory must already exist");
    assert!(
        is_lowercase_uuid_shape(&run_uuid),
        "protected B=4 decode proof requires a lowercase UUID-shaped run identifier"
    );
    let process_start_marker =
        write_phase_marker(&phase_dir, &run_uuid, 0, "process-start", test_started);
    for name in BENCHMARK_OVERRIDE_ENV {
        assert!(
            std::env::var_os(name).is_none(),
            "B=4 decode proof requires the canonical default environment; unset {name}"
        );
    }
    assert_eq!(
        DECODE_STEPS, 132,
        "proof must retain the reviewed 132-step span"
    );
    assert!(DECODE_STEPS >= 130, "proof must execute at least 130 steps");
    let mut sorted_permutation = PHYSICAL_TO_LOGICAL;
    sorted_permutation.sort_unstable();
    assert_eq!(sorted_permutation, [0, 1, 2, 3]);

    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    assert!(model.cfg.compress_ratios.contains(&4));
    assert!(model.cfg.compress_ratios.contains(&128));
    assert_eq!(model.cfg.sliding_window, 128);

    let prefixes = (0..LANES).map(prefix_tokens).collect::<Vec<_>>();
    let mut serial_caches = (0..LANES)
        .map(|logical_lane| {
            let mut cache = model
                .allocate_cache(PARITY_CAPACITY)
                .expect("allocate serial cache");
            model
                .forward_verifier_prefill(&prefixes[logical_lane], &mut cache)
                .expect("install serial prefix");
            cache
        })
        .collect::<Vec<_>>();
    let mut cohort_caches = PHYSICAL_TO_LOGICAL
        .iter()
        .map(|&logical_lane| {
            let mut cache = model
                .allocate_cache(PARITY_CAPACITY)
                .expect("allocate cohort cache");
            model
                .forward_verifier_prefill(&prefixes[logical_lane], &mut cache)
                .expect("install cohort prefix");
            cache
        })
        .collect::<Vec<_>>();
    for step in 0..DECODE_STEPS {
        let logical_tokens = supplied_tokens(step);
        let mut distinct = logical_tokens;
        distinct.sort_unstable();
        assert!(
            distinct.windows(2).all(|pair| pair[0] != pair[1]),
            "decode step {step} must use distinct lane tokens"
        );

        let mut serial_states = Vec::with_capacity(LANES);
        let mut serial_logits = Vec::with_capacity(LANES);
        for logical_lane in 0..LANES {
            let state = model
                .forward_verifier_one(
                    logical_tokens[logical_lane],
                    &mut serial_caches[logical_lane],
                )
                .unwrap_or_else(|error| {
                    panic!("serial step {step} lane {logical_lane} body: {error:#}")
                });
            let logits = model.forward_logits(&state).unwrap_or_else(|error| {
                panic!("serial step {step} lane {logical_lane} head: {error:#}")
            });
            serial_states.push(read_f32(&state, "serial decode state"));
            serial_logits.push(read_f32(&logits, "serial logits"));
        }

        let physical_tokens = PHYSICAL_TO_LOGICAL.map(|logical_lane| logical_tokens[logical_lane]);
        let [lane0, lane1, lane2, lane3] = cohort_caches.as_mut_slice() else {
            panic!("B=4 parity cache count drift")
        };
        let mut cache_refs = [lane0, lane1, lane2, lane3];
        let cohort_state = model
            .forward_verifier_decode_cohort(physical_tokens, &mut cache_refs)
            .unwrap_or_else(|error| panic!("B=4 step {step} body: {error:#}"));
        let cohort_logits = model
            .forward_logits(&cohort_state)
            .unwrap_or_else(|error| panic!("B=4 step {step} head: {error:#}"));
        assert_eq!(cohort_state.shape(), [LANES, 4, 4_096]);
        assert_eq!(
            cohort_logits.shape(),
            [LANES, model.cfg.vocab_size as usize]
        );
        let cohort_state = read_f32(&cohort_state, "B=4 decode state");
        let cohort_logits = read_f32(&cohort_logits, "B=4 logits");
        let state_row = 4 * 4_096;
        let logit_row = model.cfg.vocab_size as usize;
        for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
            assert_exact_f32(
                &format!("step {step} logical lane {logical_lane} state"),
                &serial_states[logical_lane],
                &cohort_state[physical_lane * state_row..(physical_lane + 1) * state_row],
            );
            assert_exact_f32(
                &format!("step {step} logical lane {logical_lane} logits"),
                &serial_logits[logical_lane],
                &cohort_logits[physical_lane * logit_row..(physical_lane + 1) * logit_row],
            );
            assert_cache_exact(
                &format!("step {step} logical lane {logical_lane} cache"),
                &serial_caches[logical_lane],
                &cohort_caches[physical_lane],
            );
        }
    }
    assert_eq!(serial_caches[0].position(), PARITY_CAPACITY);
    assert_eq!(cohort_caches[0].position(), PARITY_CAPACITY);
    assert_eq!(
        PARITY_CAPACITY % 128,
        24,
        "proof must cross a ratio-128 boundary"
    );
    assert_eq!(
        PARITY_CAPACITY % 4,
        0,
        "proof must finish on a ratio-4 boundary"
    );

    serial_caches = grow_and_extend_caches(&mut model, serial_caches, [0, 1, 2, 3]);
    cohort_caches = grow_and_extend_caches(&mut model, cohort_caches, PHYSICAL_TO_LOGICAL);
    for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
        assert_cache_exact(
            &format!("benchmark anchor logical lane {logical_lane} cache"),
            &serial_caches[logical_lane],
            &cohort_caches[physical_lane],
        );
    }
    let serial_snapshots = serial_caches
        .iter()
        .map(|cache| cache.snapshot().expect("snapshot serial benchmark anchor"))
        .collect::<Vec<_>>();
    let cohort_snapshots = cohort_caches
        .iter()
        .map(|cache| cache.snapshot().expect("snapshot cohort benchmark anchor"))
        .collect::<Vec<_>>();

    // The short parity span proves ratio-boundary behavior, but the benchmark
    // runs at the product recovery anchor. Prove the exact same transaction at
    // that depth before interpreting any timing from it, then restore the
    // byte-identical anchors used by every trial.
    assert_decode_step_exact(
        "benchmark anchor position 6676",
        &mut model,
        &mut serial_caches,
        &mut cohort_caches,
        supplied_tokens(DECODE_STEPS + 1),
    );
    restore_caches(
        &mut serial_caches,
        &serial_snapshots,
        "serial benchmark anchor exactness",
    );
    restore_caches(
        &mut cohort_caches,
        &cohort_snapshots,
        "cohort benchmark anchor exactness",
    );
    for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
        assert_cache_exact(
            &format!("restored benchmark anchor logical lane {logical_lane} cache"),
            &serial_caches[logical_lane],
            &cohort_caches[physical_lane],
        );
    }

    let weight_resident_bytes = model.weights.resident_bytes();
    let weight_file_backed_bytes = model.weights.file_backed_bytes();
    let weight_anonymous_bytes = model.weights.anonymous_bytes();
    let weight_mapped_segment_count = model.weights.mapped_segment_count();
    let residency_shape_pass = weight_mapped_segment_count > 0
        && weight_file_backed_bytes > weight_anonymous_bytes
        && weight_resident_bytes
            == weight_file_backed_bytes
                .checked_add(weight_anonymous_bytes)
                .expect("weight residency shape overflow");
    let serial_cache_resident_bytes = checked_resident_sum(
        "serial live caches",
        serial_caches.iter().map(Deepseek4Cache::resident_bytes),
    );
    let cohort_cache_resident_bytes = checked_resident_sum(
        "cohort live caches",
        cohort_caches.iter().map(Deepseek4Cache::resident_bytes),
    );
    let serial_snapshot_resident_bytes = checked_resident_sum(
        "serial snapshots",
        serial_snapshots
            .iter()
            .map(Deepseek4CacheSnapshot::resident_bytes),
    );
    let cohort_snapshot_resident_bytes = checked_resident_sum(
        "cohort snapshots",
        cohort_snapshots
            .iter()
            .map(Deepseek4CacheSnapshot::resident_bytes),
    );
    let tracked_resident_bytes = checked_resident_sum(
        "benchmark total",
        [
            weight_resident_bytes,
            serial_cache_resident_bytes,
            cohort_cache_resident_bytes,
            serial_snapshot_resident_bytes,
            cohort_snapshot_resident_bytes,
        ],
    );
    eprintln!(
        "DeepSeek-V4 B=4 benchmark loaded-idle settle: position={BENCHMARK_POSITION} logical_capacity={BENCHMARK_LOGICAL_CAPACITY} seconds={LOADED_IDLE_SECONDS} weight_resident_bytes={weight_resident_bytes} serial_cache_resident_bytes={serial_cache_resident_bytes} cohort_cache_resident_bytes={cohort_cache_resident_bytes} serial_snapshot_resident_bytes={serial_snapshot_resident_bytes} cohort_snapshot_resident_bytes={cohort_snapshot_resident_bytes} tracked_resident_bytes={tracked_resident_bytes}"
    );
    let loaded_settle_marker = write_phase_marker(
        &phase_dir,
        &run_uuid,
        1,
        "loaded-settle-start",
        test_started,
    );
    std::thread::sleep(Duration::from_secs(LOADED_IDLE_SECONDS));
    for (cache, snapshot) in serial_caches.iter_mut().zip(&serial_snapshots) {
        cache
            .restore(snapshot)
            .expect("restore serial warmup cache");
    }
    let warm_tokens = supplied_tokens(0);
    let _ = timed_serial(&mut model, &mut serial_caches, warm_tokens);
    for (cache, snapshot) in cohort_caches.iter_mut().zip(&cohort_snapshots) {
        cache
            .restore(snapshot)
            .expect("restore cohort warmup cache");
    }
    let _ = timed_cohort(&mut model, &mut cohort_caches, warm_tokens);
    let measurement_ready_marker =
        write_phase_marker(&phase_dir, &run_uuid, 2, "measurement-ready", test_started);
    wait_for_measurement_ack(&phase_dir, &run_uuid);
    let vm_window_start = capture_darwin_vm_snapshot();

    let mut serial_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut cohort_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut serial_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut cohort_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    for pair in 0..BENCHMARK_PAIRS {
        let tokens = supplied_tokens(pair + 1);
        if pair % 2 == 0 {
            for (cache, snapshot) in serial_caches.iter_mut().zip(&serial_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore serial timing cache");
            }
            let (elapsed, counters) = timed_serial(&mut model, &mut serial_caches, tokens);
            serial_ms.push(elapsed);
            serial_counters.push(counters);
            for (cache, snapshot) in cohort_caches.iter_mut().zip(&cohort_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore cohort timing cache");
            }
            let (elapsed, counters) = timed_cohort(&mut model, &mut cohort_caches, tokens);
            cohort_ms.push(elapsed);
            cohort_counters.push(counters);
        } else {
            for (cache, snapshot) in cohort_caches.iter_mut().zip(&cohort_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore cohort timing cache");
            }
            let (elapsed, counters) = timed_cohort(&mut model, &mut cohort_caches, tokens);
            cohort_ms.push(elapsed);
            cohort_counters.push(counters);
            for (cache, snapshot) in serial_caches.iter_mut().zip(&serial_snapshots) {
                cache
                    .restore(snapshot)
                    .expect("restore serial timing cache");
            }
            let (elapsed, counters) = timed_serial(&mut model, &mut serial_caches, tokens);
            serial_ms.push(elapsed);
            serial_counters.push(counters);
        }
    }

    // Diagnostic control for topology-conditioned transient lifetime work.
    // Every measured arm follows an untimed execution of that same arm from
    // the exact benchmark snapshot. The second restore keeps the measured
    // cache/token work identical while preventing a serial trial from
    // inheriting cohort pool geometry (and vice versa).
    let mut conditioned_serial_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_serial_prime_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_prime_ms = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_serial_prime_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_prime_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_serial_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    let mut conditioned_cohort_counters = Vec::with_capacity(BENCHMARK_PAIRS);
    for pair in 0..BENCHMARK_PAIRS {
        let tokens = supplied_tokens(pair + 1);
        if pair % 2 == 0 {
            let (prime_elapsed, prime_counters, elapsed, counters) =
                timed_serial_conditioned(&mut model, &mut serial_caches, &serial_snapshots, tokens);
            conditioned_serial_prime_ms.push(prime_elapsed);
            conditioned_serial_prime_counters.push(prime_counters);
            conditioned_serial_ms.push(elapsed);
            conditioned_serial_counters.push(counters);
            let (prime_elapsed, prime_counters, elapsed, counters) =
                timed_cohort_conditioned(&mut model, &mut cohort_caches, &cohort_snapshots, tokens);
            conditioned_cohort_prime_ms.push(prime_elapsed);
            conditioned_cohort_prime_counters.push(prime_counters);
            conditioned_cohort_ms.push(elapsed);
            conditioned_cohort_counters.push(counters);
        } else {
            let (prime_elapsed, prime_counters, elapsed, counters) =
                timed_cohort_conditioned(&mut model, &mut cohort_caches, &cohort_snapshots, tokens);
            conditioned_cohort_prime_ms.push(prime_elapsed);
            conditioned_cohort_prime_counters.push(prime_counters);
            conditioned_cohort_ms.push(elapsed);
            conditioned_cohort_counters.push(counters);
            let (prime_elapsed, prime_counters, elapsed, counters) =
                timed_serial_conditioned(&mut model, &mut serial_caches, &serial_snapshots, tokens);
            conditioned_serial_prime_ms.push(prime_elapsed);
            conditioned_serial_prime_counters.push(prime_counters);
            conditioned_serial_ms.push(elapsed);
            conditioned_serial_counters.push(counters);
        }
    }

    let vm_window_end = capture_darwin_vm_snapshot();
    let measurement_complete_marker = write_phase_marker(
        &phase_dir,
        &run_uuid,
        3,
        "measurement-complete",
        test_started,
    );

    assert_eq!(serial_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(cohort_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(conditioned_serial_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(conditioned_cohort_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(conditioned_serial_prime_ms.len(), BENCHMARK_PAIRS);
    assert_eq!(conditioned_cohort_prime_ms.len(), BENCHMARK_PAIRS);
    let hidden_layers =
        u64::try_from(model.cfg.num_hidden_layers).expect("DeepSeek-V4 layer count must fit u64");
    let body_command_buffers = hidden_layers.div_ceil(2);
    let expected_serial_command_buffers = LANES as u64 * (body_command_buffers + 1);
    let expected_cohort_command_buffers = body_command_buffers + 1;
    let mut topology_errors = Vec::new();
    for pair in 0..BENCHMARK_PAIRS {
        record_counter_errors(
            &mut topology_errors,
            &format!("pair {pair} unconditioned serial"),
            serial_counters[pair],
            expected_serial_command_buffers,
            LANES as u64,
        );
        record_counter_errors(
            &mut topology_errors,
            &format!("pair {pair} unconditioned cohort"),
            cohort_counters[pair],
            expected_cohort_command_buffers,
            1,
        );
        record_counter_errors(
            &mut topology_errors,
            &format!("pair {pair} conditioned serial prime"),
            conditioned_serial_prime_counters[pair],
            expected_serial_command_buffers,
            LANES as u64,
        );
        record_counter_errors(
            &mut topology_errors,
            &format!("pair {pair} conditioned serial measure"),
            conditioned_serial_counters[pair],
            expected_serial_command_buffers,
            LANES as u64,
        );
        record_counter_errors(
            &mut topology_errors,
            &format!("pair {pair} conditioned cohort prime"),
            conditioned_cohort_prime_counters[pair],
            expected_cohort_command_buffers,
            1,
        );
        record_counter_errors(
            &mut topology_errors,
            &format!("pair {pair} conditioned cohort measure"),
            conditioned_cohort_counters[pair],
            expected_cohort_command_buffers,
            1,
        );
    }
    let topology_pass = topology_errors.is_empty();
    let serial_median = median(serial_ms.clone());
    let cohort_median = median(cohort_ms.clone());
    let speedup = serial_median / cohort_median;
    let conditioned_serial_median = median(conditioned_serial_ms.clone());
    let conditioned_cohort_median = median(conditioned_cohort_ms.clone());
    let conditioned_serial_even_median = median(order_stratum(&conditioned_serial_ms, 0));
    let conditioned_cohort_even_median = median(order_stratum(&conditioned_cohort_ms, 0));
    let conditioned_serial_odd_median = median(order_stratum(&conditioned_serial_ms, 1));
    let conditioned_cohort_odd_median = median(order_stratum(&conditioned_cohort_ms, 1));
    let conditioned_speedup = conditioned_serial_median / conditioned_cohort_median;
    let conditioned_even_speedup = conditioned_serial_even_median / conditioned_cohort_even_median;
    let conditioned_odd_speedup = conditioned_serial_odd_median / conditioned_cohort_odd_median;
    let conditioned_deltas = paired_deltas(&conditioned_serial_ms, &conditioned_cohort_ms);
    let conditioned_even_delta_median = median(order_stratum(&conditioned_deltas, 0));
    let conditioned_odd_delta_median = median(order_stratum(&conditioned_deltas, 1));
    let unconditioned_deltas = paired_deltas(&serial_ms, &cohort_ms);
    let unconditioned_even_delta_median = median(order_stratum(&unconditioned_deltas, 0));
    let unconditioned_odd_delta_median = median(order_stratum(&unconditioned_deltas, 1));
    let unconditioned_order_signature =
        unconditioned_even_delta_median < 0.0 && unconditioned_odd_delta_median > 0.0;
    let vm_monotonic = vm_window_end.pageins >= vm_window_start.pageins
        && vm_window_end.pageouts >= vm_window_start.pageouts
        && vm_window_end.swapins >= vm_window_start.swapins
        && vm_window_end.swapouts >= vm_window_start.swapouts
        && vm_window_end.compressions >= vm_window_start.compressions
        && vm_window_end.decompressions >= vm_window_start.decompressions
        && vm_window_end.purges >= vm_window_start.purges
        && vm_window_end.reactivations >= vm_window_start.reactivations
        && vm_window_end.process_pageins >= vm_window_start.process_pageins;
    let pagein_delta = vm_window_end
        .pageins
        .saturating_sub(vm_window_start.pageins);
    let pageout_delta = vm_window_end
        .pageouts
        .saturating_sub(vm_window_start.pageouts);
    let swapin_delta = vm_window_end
        .swapins
        .saturating_sub(vm_window_start.swapins);
    let swapout_delta = vm_window_end
        .swapouts
        .saturating_sub(vm_window_start.swapouts);
    let compression_delta = vm_window_end
        .compressions
        .saturating_sub(vm_window_start.compressions);
    let decompression_delta = vm_window_end
        .decompressions
        .saturating_sub(vm_window_start.decompressions);
    let purge_delta = vm_window_end.purges.saturating_sub(vm_window_start.purges);
    let reactivation_delta = vm_window_end
        .reactivations
        .saturating_sub(vm_window_start.reactivations);
    let process_pagein_delta = vm_window_end
        .process_pageins
        .saturating_sub(vm_window_start.process_pageins);
    let pressure_boundary_pass = matches!(vm_window_start.pressure_level, 1 | 2)
        && matches!(vm_window_end.pressure_level, 1 | 2);
    let vm_window_pass = vm_monotonic
        && vm_window_start.boot_time_seconds == vm_window_end.boot_time_seconds
        && vm_window_start.page_size == vm_window_end.page_size
        && pressure_boundary_pass
        && vm_window_start.throttled_pages == 0
        && vm_window_end.throttled_pages == 0
        && pagein_delta == 0
        && pageout_delta == 0
        && swapin_delta == 0
        && swapout_delta == 0
        && compression_delta == 0
        && decompression_delta == 0
        && purge_delta == 0
        && process_pagein_delta == 0;
    eprintln!(
        "DeepSeek-V4 B=4 decode spike: artifact={} parity_prefix={} parity_steps={} benchmark_position={} benchmark_logical_capacity={} loaded_idle_seconds={} permutation={:?} pairs={} order=alternating exact_state_logits_cache_recurrent=true benchmark_anchor_exact_state_logits_cache_recurrent=true tracked_resident_bytes={} vm_window_pass={} pagein_delta={} pageout_delta={} swapin_delta={} swapout_delta={} compression_delta={} decompression_delta={} purge_delta={} process_pagein_delta={} serial_ms={:?} cohort_ms={:?} serial_median_ms={:.3} cohort_median_ms={:.3} speedup={:.4}x unconditioned_even_delta_median_ms={:.3} unconditioned_odd_delta_median_ms={:.3} unconditioned_order_signature={} unconditioned_is_diagnostic_only=true conditioned_serial_prime_ms={:?} conditioned_cohort_prime_ms={:?} conditioned_serial_ms={:?} conditioned_cohort_ms={:?} conditioned_serial_median_ms={:.3} conditioned_cohort_median_ms={:.3} conditioned_speedup={:.4}x conditioned_even_speedup={:.4}x conditioned_odd_speedup={:.4}x conditioned_even_delta_median_ms={:.3} conditioned_odd_delta_median_ms={:.3} topology_pass={} topology_errors={:?} serial_counters={:?} cohort_counters={:?} conditioned_serial_prime_counters={:?} conditioned_cohort_prime_counters={:?} conditioned_serial_counters={:?} conditioned_cohort_counters={:?}",
        path.display(),
        PREFIX_ROWS,
        DECODE_STEPS,
        BENCHMARK_POSITION,
        BENCHMARK_LOGICAL_CAPACITY,
        LOADED_IDLE_SECONDS,
        PHYSICAL_TO_LOGICAL,
        BENCHMARK_PAIRS,
        tracked_resident_bytes,
        vm_window_pass,
        pagein_delta,
        pageout_delta,
        swapin_delta,
        swapout_delta,
        compression_delta,
        decompression_delta,
        purge_delta,
        process_pagein_delta,
        serial_ms,
        cohort_ms,
        serial_median,
        cohort_median,
        speedup,
        unconditioned_even_delta_median,
        unconditioned_odd_delta_median,
        unconditioned_order_signature,
        conditioned_serial_prime_ms,
        conditioned_cohort_prime_ms,
        conditioned_serial_ms,
        conditioned_cohort_ms,
        conditioned_serial_median,
        conditioned_cohort_median,
        conditioned_speedup,
        conditioned_even_speedup,
        conditioned_odd_speedup,
        conditioned_even_delta_median,
        conditioned_odd_delta_median,
        topology_pass,
        topology_errors,
        serial_counters,
        cohort_counters,
        conditioned_serial_prime_counters,
        conditioned_cohort_prime_counters,
        conditioned_serial_counters,
        conditioned_cohort_counters,
    );
    let conditioned_performance_pass = conditioned_serial_median.is_finite()
        && conditioned_cohort_median.is_finite()
        && conditioned_serial_median > conditioned_cohort_median
        && conditioned_cohort_median > 0.0
        && conditioned_even_speedup > 1.0
        && conditioned_odd_speedup > 1.0
        && conditioned_even_delta_median > 0.0
        && conditioned_odd_delta_median > 0.0;
    let conditioned_pass =
        conditioned_performance_pass && topology_pass && vm_window_pass && residency_shape_pass;

    let artifact_bytes = std::fs::metadata(&path)
        .unwrap_or_else(|error| panic!("stat official artifact {}: {error}", path.display()))
        .len();
    let counters_json = |counters: &[TrialCounters]| {
        counters
            .iter()
            .map(|counter| {
                serde_json::json!({
                    "command_buffers": counter.command_buffers,
                    "synchronizations": counter.synchronizations,
                    "dispatches": counter.dispatches,
                    "barriers": counter.barriers,
                })
            })
            .collect::<Vec<_>>()
    };
    let residency_json = serde_json::json!({
        "effective_weight_mode": "mmap-file-backed",
        "weight_bytes": weight_resident_bytes,
        "weight_file_backed_bytes": weight_file_backed_bytes,
        "weight_anonymous_bytes": weight_anonymous_bytes,
        "weight_mapped_segment_count": weight_mapped_segment_count,
        "shape_pass": residency_shape_pass,
        "serial_live_cache_bytes": serial_cache_resident_bytes,
        "cohort_live_cache_bytes": cohort_cache_resident_bytes,
        "serial_snapshot_bytes": serial_snapshot_resident_bytes,
        "cohort_snapshot_bytes": cohort_snapshot_resident_bytes,
        "tracked_total_bytes": tracked_resident_bytes,
    });
    let phase_contract_json = serde_json::json!({
        "policy": "fsynced-run-bound-markers-v1",
        "run_uuid": run_uuid,
        "pid": std::process::id(),
        "ack_timeout_seconds": PHASE_ACK_TIMEOUT_SECONDS,
        "markers": [
            process_start_marker.json(),
            loaded_settle_marker.json(),
            measurement_ready_marker.json(),
            measurement_complete_marker.json(),
        ],
    });
    let vm_window_json = serde_json::json!({
        "policy": "darwin25-phase-bound-no-vm-churn-v2",
        "claim_scope": "within-run-paired-only",
        "start": vm_window_start.json(),
        "end": vm_window_end.json(),
        "deltas": {
            "pageins": pagein_delta,
            "pageouts": pageout_delta,
            "swapins": swapin_delta,
            "swapouts": swapout_delta,
            "compressions": compression_delta,
            "decompressions": decompression_delta,
            "purges": purge_delta,
            "reactivations": reactivation_delta,
            "process_pageins": process_pagein_delta,
        },
        "monotonic_counters": vm_monotonic,
        "pressure_boundary_pass": pressure_boundary_pass,
        "no_churn_pass": vm_window_pass,
    });
    let conditioned_json = serde_json::json!({
        "protocol": "same-topology-prime-restore-measure",
        "primes_per_measurement": 1,
        "serial_prime_ms": conditioned_serial_prime_ms,
        "cohort_prime_ms": conditioned_cohort_prime_ms,
        "serial_ms": conditioned_serial_ms,
        "cohort_ms": conditioned_cohort_ms,
        "serial_median_ms": conditioned_serial_median,
        "cohort_median_ms": conditioned_cohort_median,
        "speedup": conditioned_speedup,
        "even_order_speedup": conditioned_even_speedup,
        "odd_order_speedup": conditioned_odd_speedup,
        "even_order_paired_delta_median_ms": conditioned_even_delta_median,
        "odd_order_paired_delta_median_ms": conditioned_odd_delta_median,
        "performance_pass": conditioned_performance_pass,
        "serial_prime_counters": counters_json(&conditioned_serial_prime_counters),
        "cohort_prime_counters": counters_json(&conditioned_cohort_prime_counters),
        "serial_counters": counters_json(&conditioned_serial_counters),
        "cohort_counters": counters_json(&conditioned_cohort_counters),
    });
    let benchmark_json = serde_json::json!({
        "position": BENCHMARK_POSITION,
        "anchor_exact_state_logits_cache_recurrent": true,
        "logical_capacity": BENCHMARK_LOGICAL_CAPACITY,
        "loaded_idle_seconds": LOADED_IDLE_SECONDS,
        "pairs": BENCHMARK_PAIRS,
        "order": "alternating",
        "serial_ms": serial_ms,
        "cohort_ms": cohort_ms,
        "serial_median_ms": serial_median,
        "cohort_median_ms": cohort_median,
        "speedup": speedup,
        "unconditioned_order_signature": {
            "historical_signature": "even_delta_negative_odd_delta_positive",
            "even_delta_median_ms": unconditioned_even_delta_median,
            "odd_delta_median_ms": unconditioned_odd_delta_median,
            "observed": unconditioned_order_signature,
            "gating": false,
        },
        "conditioned": conditioned_json,
        "serial_counters": counters_json(&serial_counters),
        "cohort_counters": counters_json(&cohort_counters),
        "serial_command_buffers_per_pair": expected_serial_command_buffers,
        "cohort_command_buffers_per_pair": expected_cohort_command_buffers,
        "serial_synchronizations_per_pair": LANES,
        "cohort_synchronizations_per_pair": 1,
        "topology_pass": topology_pass,
        "topology_errors": topology_errors,
    });
    let receipt = serde_json::json!({
            "schema_version": 2,
            "status": if conditioned_pass { "pass" } else { "fail" },
            "artifact_bytes": artifact_bytes,
            "layers": model.cfg.num_hidden_layers,
            "lanes": LANES,
            "parity": {
                "prefix_rows": PREFIX_ROWS,
                "steps": DECODE_STEPS,
                "final_position": PARITY_CAPACITY,
                "final_mod_4": PARITY_CAPACITY % 4,
                "final_mod_128": PARITY_CAPACITY % 128,
                "physical_to_logical": PHYSICAL_TO_LOGICAL,
                "exact_state_logits_cache_recurrent": true,
            },
            "residency": residency_json,
            "phase_contract": phase_contract_json,
            "darwin_vm_window": vm_window_json,
            "benchmark": benchmark_json,
            "benchmark_environment": {
                "profile": "clean-hf2q-mlx-metal-v1",
                "override_variables_absent": true,
                "unexpected_override_variables": [],
            },
    });
    let parent = receipt_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent).unwrap_or_else(|error| {
        panic!(
            "create B=4 decode receipt directory {}: {error}",
            parent.display()
        )
    });
    let temporary = receipt_path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(&receipt).expect("serialize B=4 decode receipt"),
    )
    .unwrap_or_else(|error| panic!("write B=4 decode receipt {}: {error}", temporary.display()));
    std::fs::rename(&temporary, &receipt_path).unwrap_or_else(|error| {
        panic!(
            "publish B=4 decode receipt {} -> {}: {error}",
            temporary.display(),
            receipt_path.display()
        )
    });
    assert!(
        conditioned_pass,
        "exact B=4 conditioned benchmark must retain residency/topology, zero VM churn, and a positive overall/even/odd paired median: residency_shape_pass={residency_shape_pass} vm_window_pass={vm_window_pass} topology_errors={topology_errors:?} serial={conditioned_serial_median:.3}ms cohort={conditioned_cohort_median:.3}ms even={conditioned_even_speedup:.4}x/{conditioned_even_delta_median:.3}ms odd={conditioned_odd_speedup:.4}x/{conditioned_odd_delta_median:.3}ms"
    );
}
