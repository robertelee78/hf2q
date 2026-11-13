//! Real-artifact observation of an engine's teardown.
//!
//! Loads a Qwen model through `serve::load_engine` on the default route,
//! drops the engine as an eviction does (nobody stops or joins the worker),
//! and reads the process's physical footprint at the moment the process
//! ledger lets the reservation go, and for six seconds after. It checks that
//! the teardown the code performs (model dropped, thread caches dropped,
//! thread terminated, queues settled, kernel given time) leaves nothing that
//! comes back later, on this host and this model. It is an observation of
//! the whole process, not a proof that an engine's bytes are back: what
//! keeps a load from being admitted to memory that is not back is the
//! present bound of the admission (`serve::resident_memory`). Run alone, on
//! a release build:
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
fn nothing_of_a_real_engine_comes_back_after_its_reservation_is_released() {
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
    let mut still_resident = Vec::new();
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
        // What the footprint does after the release: nothing more may come
        // back, or the reservation went before the memory did.
        let mut settled = released;
        let mut later = Vec::new();
        for wait_ms in [10_u64, 100, 1_000, 5_000] {
            std::thread::sleep(Duration::from_millis(wait_ms));
            let footprint = footprint_bytes();
            settled = settled.min(footprint);
            later.push(format!(
                "+{}ms={:.2}",
                evicted.elapsed().as_millis() - teardown.as_millis(),
                footprint as f64 / GIB
            ));
        }
        let held = loaded.saturating_sub(baseline);
        let returned_late = released - settled;
        println!(
            "[LIFECYCLE] round={round} selected={} profile={} available_gib={:.2} \
             committed_gib={:.2} loaded_gib={:.2} at_release_gib={:.2} \
             after_release_gib=[{}] returned_after_release_gib={:.3} teardown_ms={}",
            decision["selected"],
            decision["profile"],
            decision["memory"]["available_bytes"].as_u64().unwrap_or(0) as f64 / GIB,
            committed as f64 / GIB,
            loaded as f64 / GIB,
            released as f64 / GIB,
            later.join(" "),
            returned_late as f64 / GIB,
            teardown.as_millis()
        );
        assert!(
            held as f64 > 0.5 * std::fs::metadata(&path).expect("model").len() as f64,
            "round {round}: the loaded engine held only {held} bytes"
        );
        assert!(
            (released.saturating_sub(baseline) as f64) < 0.25 * held as f64,
            "round {round}: most of the engine was never returned"
        );
        // What stays after a teardown is the allocator's cache of freed host
        // blocks (`Malloc Large (empty)` in vmmap), which this process can
        // reuse. Nothing of the engine was seen to come back after the
        // release.
        if returned_late > 256 << 20 {
            still_resident.push(format!(
                "round {round}: {returned_late} bytes came back after the reservation was \
                 released"
            ));
        }
    }
    assert!(still_resident.is_empty(), "{still_resident:#?}");
}
