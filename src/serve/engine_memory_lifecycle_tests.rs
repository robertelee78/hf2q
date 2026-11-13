//! Real-artifact proof that an engine's memory reservation is released only
//! when its memory is free.
//!
//! Loads a Qwen model through `serve::load_engine` on the default route,
//! drops the engine as an eviction does (nobody stops or joins the worker),
//! and reads the process's physical footprint at the moment the process
//! ledger lets the reservation go. Run alone, on a release build:
//!
//! ```text
//! HF2Q_ENGINE_LIFECYCLE_GGUF=/path/model.gguf cargo test --release --locked --bin hf2q -- \
//!     engine_memory_lifecycle --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::api::engine::EngineMode;
use super::{load_engine, multi_model, resident_memory};

const GIB: f64 = (1_u64 << 30) as f64;

/// The process's physical footprint, as Activity Monitor and `vmmap` count
/// it.
fn footprint_bytes() -> u64 {
    let mut info = unsafe { std::mem::zeroed::<libc::rusage_info_v4>() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        )
    };
    assert_eq!(rc, 0, "proc_pid_rusage(RUSAGE_INFO_V4) failed");
    info.ri_phys_footprint
}

#[test]
#[ignore = "loads a real model twice; set HF2Q_ENGINE_LIFECYCLE_GGUF"]
fn a_real_engines_reservation_is_released_only_when_its_memory_is_free() {
    let path = PathBuf::from(
        std::env::var("HF2Q_ENGINE_LIFECYCLE_GGUF").expect("set HF2Q_ENGINE_LIFECYCLE_GGUF"),
    );
    let config = multi_model::EngineConfig {
        operator_model_path: None,
        tokenizer_path: None,
        config_path: None,
        queue_capacity: 8,
        // The warmup makes the worker build its GPU weight cache: the
        // copies and planes the teardown has to give back.
        warmup_synchronously: true,
        glp_reference: None,
        glp_alpha: None,
        kv_graft_path: None,
        lcp_resume_budget: crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT,
        kv_metrics_sink: None,
        dwq_overlay_path: None,
        engine_mode: EngineMode::SlotAware { max_slots: 4 },
        requested_context: None,
        kv_cache_budget_bytes: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        slot_spill_budget_bytes: None,
    };
    let ledger = resident_memory::process();
    assert_eq!(ledger.total(), 0, "run this test alone");
    let baseline = footprint_bytes();
    for round in 1..=2 {
        let engine = load_engine(&path, &config).expect("load");
        let decision = engine.route_decision().expect("route decision");
        let committed = ledger.total();
        let loaded = footprint_bytes();
        assert!(committed > 0);
        // A load after an engine was torn down is admitted against nothing.
        if decision["selected"] == true {
            assert_eq!(decision["memory"]["resident_bytes"], 0, "round {round}");
        }

        // Evicted: the last handle goes. The worker is not stopped by
        // anyone and not waited for.
        let evicted = Instant::now();
        drop(engine);
        while ledger.total() != 0 {
            assert!(
                evicted.elapsed() < Duration::from_secs(120),
                "round {round}: the worker did not tear down"
            );
            std::thread::sleep(Duration::from_micros(200));
        }
        let teardown = evicted.elapsed();
        let released = footprint_bytes();

        let held = loaded.saturating_sub(baseline);
        let left = released.saturating_sub(baseline);
        println!(
            "[LIFECYCLE] round={round} selected={} profile={} committed_gib={:.2} \
             baseline_gib={:.2} loaded_gib={:.2} at_release_gib={:.2} held_gib={:.2} \
             left_gib={:.2} teardown_ms={}",
            decision["selected"],
            decision["profile"],
            committed as f64 / GIB,
            baseline as f64 / GIB,
            loaded as f64 / GIB,
            released as f64 / GIB,
            held as f64 / GIB,
            left as f64 / GIB,
            teardown.as_millis()
        );
        assert!(
            held as f64 > 0.5 * std::fs::metadata(&path).expect("model").len() as f64,
            "round {round}: the loaded engine held only {held} bytes"
        );
        // When the ledger says the memory is free, it is: what is left of
        // the engine is a small fraction of what it held.
        assert!(
            (left as f64) < 0.05 * held as f64,
            "round {round}: {left} of {held} bytes were still resident when the reservation \
             was released"
        );
    }
}
