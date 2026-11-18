//! Explicit fresh-process diagnostic; never linked into the production CLI.
//! The outer canonical Q5 supervisor still owns physical-memory safety.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use mlx_native::{
    LiveRootLimits, MlxDevice, NativeRootAuthority, RootAuthorityHandle, RootCreationLimits,
};
use serde_json::{json, Value};

fn observation(authority: RootAuthorityHandle) -> Value {
    match (authority.snapshot(), authority.live_snapshot()) {
        (Ok(history), Ok(live)) => json!({
            "scope": live.scope,
            "covered": history.covered && !history.legacy_used && history.unknown.is_none()
                && live.covered && live.unknown.is_none(),
            "physical_residency_claim": false,
            "history": {
                "attempted_requested_bytes": history.attempted_requested_bytes,
                "root_attempts": history.root_attempts,
                "raw_attempt_reservations": history.raw_attempt_reservations,
                "successful_roots": history.successful_roots,
                "inflight": history.inflight,
                "registered_devices": history.registered_devices,
                "published_cache_entries": history.published_cache_entries,
                "denied": history.denied,
                "unknown": history.unknown,
                "legacy_used": history.legacy_used,
            },
            "live": {
                "pending_requested_bytes": live.pending_requested_bytes,
                "live_requested_bytes": live.live_requested_bytes,
                "pending_roots": live.pending_roots,
                "live_roots": live.live_roots,
                "high_water_requested_bytes": live.high_water_requested_bytes,
                "final_releases": live.final_releases,
                "denied": live.denied,
                "unknown": live.unknown,
            },
        }),
        (history, live) => json!({
            "covered": false,
            "physical_residency_claim": false,
            "history_error": history.err().map(|error| error.to_string()),
            "live_error": live.err().map(|error| error.to_string()),
        }),
    }
}

fn record(output: &Mutex<File>, started: Instant, phase: &str, snapshot: &Value) {
    let mut file = output.lock().expect("probe output lock");
    serde_json::to_writer(
        &mut *file,
        &json!({
            "schema": "hf2q.private.native-requested-root-lifecycle.v1",
            "pid": std::process::id(),
            "elapsed_ns": started.elapsed().as_nanos(),
            "phase": phase,
            "snapshot": snapshot,
        }),
    )
    .expect("write complete probe record");
    file.write_all(b"\n").expect("terminate probe record");
    file.flush().expect("flush probe record");
}

#[test]
#[ignore = "fresh process only; explicit canonical Q5 supervisor, argv and output required; actual model work"]
fn canonical_n4_requested_roots() {
    let raw = std::env::var("HF2Q_ROOT_PROBE_ARGV").expect("explicit probe argv");
    assert!(raw.len() <= 96 * 1024, "bounded probe argv");
    let args: Vec<String> = serde_json::from_str(&raw).expect("probe argv JSON");
    assert!(!args.is_empty() && args.len() <= 256);
    assert!(args.iter().all(|arg| arg.len() <= 4096));
    let cli = crate::cli::Cli::try_parse_from(&args).expect("actual served CLI arguments");
    assert!(
        matches!(&cli.command, crate::cli::Command::Serve(_)),
        "serve only"
    );
    let output_path = std::env::var_os("HF2Q_ROOT_PROBE_OUTPUT").expect("explicit probe output");
    let output = Arc::new(Mutex::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output_path)
            .expect("exclusive probe output"),
    ));
    let started = Instant::now();
    // Historical/requested ceilings only. They do not replace or reduce any
    // canonical admission reserve, process cap, pressure or swap safety rule.
    let authority = NativeRootAuthority::install_early_with_live(
        RootCreationLimits {
            max_attempted_requested_bytes: 1usize << 40,
            max_root_attempts: 1_000_000,
            max_raw_constructor_attempts: 1_000_000,
            max_registered_devices: 32,
            max_rope_cache_entries: 4096,
        },
        LiveRootLimits {
            max_live_and_pending_requested_bytes: 128usize << 30,
            max_live_and_pending_roots: 100_000,
        },
    )
    .expect("fresh native requested-root authority");
    let seed = MlxDevice::new().expect("initial device without root creation");
    authority
        .register_device(&seed)
        .expect("register actual default Metal device");
    record(&output, started, "installed", &observation(authority));

    crate::configure_native_xet_performance_policy();
    crate::debug::INVESTIGATION_ENV.activate(matches!(cli.log_format, crate::cli::LogFormat::Json));
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "hf2q=info,mlx_native=info",
        ))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .without_time()
        .try_init()
        .expect("fresh diagnostic logging subscriber");
    let stop = Arc::new(AtomicBool::new(false));
    let sampling_stop = Arc::clone(&stop);
    let sampling_output = Arc::clone(&output);
    let sampler = std::thread::spawn(move || {
        for _ in 0..1024 {
            if sampling_stop.load(Ordering::Acquire) {
                return;
            }
            record(
                &sampling_output,
                started,
                "running",
                &observation(authority),
            );
            std::thread::sleep(Duration::from_millis(500));
        }
        panic!("bounded diagnostic record quota exhausted");
    });
    // Real dispatch: the existing server, model loader, scheduler, HTTP/SSE,
    // tool continuation and shutdown paths run unchanged in this process.
    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::run(cli)));
    stop.store(true, Ordering::Release);
    let sampled = sampler.join();
    drop(seed);
    let retire_deadline = Instant::now() + Duration::from_secs(2);
    let mut final_snapshot = loop {
        let snapshot = observation(authority);
        // Native rope_freqs.rs retains one distinct root per published
        // immutable process-cache entry. This is an owner-count contract,
        // not a golden for the observed cache byte size.
        let closed = snapshot["covered"] == true
            && snapshot["live"]["pending_roots"] == 0
            && snapshot["live"]["live_roots"] == snapshot["history"]["published_cache_entries"];
        if closed || snapshot["covered"] != true || Instant::now() >= retire_deadline {
            break snapshot;
        }
        record(&output, started, "retiring", &snapshot);
        std::thread::sleep(Duration::from_millis(100));
    };
    final_snapshot["retained_root_attribution"] = json!({
        "method": "source-bound immutable RoPE cache owner-count inference",
        "individual_root_identity_available": false,
        "cache_bytes_remain_accounted": true,
        "non_process_roots_retired": final_snapshot["covered"] == true
            && final_snapshot["live"]["pending_roots"] == 0
            && final_snapshot["live"]["live_roots"] == final_snapshot["history"]["published_cache_entries"],
    });
    record(
        &output,
        started,
        "after_server_and_worker_teardown",
        &final_snapshot,
    );
    sampled.expect("bounded sampler completed");
    served
        .expect("server dispatch did not unwind")
        .expect("real served command completed");
    assert_eq!(
        final_snapshot["covered"], true,
        "unavailable coverage is not a pass"
    );
    assert!(
        final_snapshot["history"]["successful_roots"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(final_snapshot["history"]["inflight"], 0);
    assert_eq!(final_snapshot["history"]["denied"], 0);
    assert_eq!(final_snapshot["live"]["denied"], 0);
    assert_eq!(final_snapshot["live"]["pending_roots"], 0);
    assert_eq!(
        final_snapshot["live"]["live_roots"], final_snapshot["history"]["published_cache_entries"],
        "roots beyond the source-bound process cache remain accounted, not credited",
    );
    assert_eq!(final_snapshot["live"]["pending_requested_bytes"], 0);
    if final_snapshot["history"]["published_cache_entries"] == 0 {
        assert_eq!(final_snapshot["live"]["live_requested_bytes"], 0);
    } else {
        assert!(
            final_snapshot["live"]["live_requested_bytes"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
}
