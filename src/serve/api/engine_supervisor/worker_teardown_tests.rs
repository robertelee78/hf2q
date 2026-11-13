//! The memory reservation of a model follows the teardown of the worker
//! thread that owns it ([`WorkerTeardown`]). Model-free: the worker body, the
//! thread's caches and the flush are stand-ins that block where a real
//! teardown can block, and the loads that compete for the memory are decided
//! by the real DFlash2 policy against the real ledger.

use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use super::{spawn_supervised_worker, DeferredFlush, EngineSupervisor, WorkerTeardown};
use crate::inference::spec_decode::dflash2::route_profile::PlaneSet;
use crate::inference::spec_decode::dflash2::schema::QWEN38_DFLASH2;
use crate::serve::dflash2_policy::tests::{env, qwen38, Store};
use crate::serve::dflash2_policy::{
    decide, plane_bytes, HostFacts, Selection, TargetFacts, MEMORY_RESERVE_BYTES,
};
use crate::serve::resident_memory::{Ledger, Reservation};

const GIB: u64 = 1 << 30;
const KV: u64 = 4 * GIB;

/// A host on which one Qwen3.8 gets its whole profile, and a second one gets
/// its planes only once the first is gone.
struct Host {
    facts: TargetFacts,
    physical: u64,
    one_engine: u64,
    ledger: Arc<Ledger>,
}

impl Host {
    fn new() -> Self {
        let facts = qwen38();
        let planes = plane_bytes(
            &facts.tensors,
            &[PlaneSet::DenseFfn, PlaneSet::Projections, PlaneSet::Head],
        );
        let drafter = QWEN38_DFLASH2.weights_bytes;
        let one_engine = facts.model_file_bytes + KV + drafter + planes;
        Self {
            // Beside a resident engine: room for a second model, its KV
            // budget and its drafter, one byte short of its planes.
            physical: MEMORY_RESERVE_BYTES + 2 * one_engine - 1,
            one_engine,
            facts,
            ledger: Ledger::new(),
        }
    }

    /// What `serve::load_engine` does before it loads: decide against what
    /// is resident and reserve the decision, as one transaction.
    fn load(&self) -> (Selection, Reservation) {
        self.ledger
            .admit(Path::new("/second.gguf"), |resident_bytes| {
                let selection = decide(
                    &self.facts,
                    HostFacts {
                        physical_bytes: self.physical,
                        kv_budget_bytes: KV,
                        resident_bytes,
                    },
                    &env(&[]),
                    &Store::cached(),
                    &|_| None,
                )?;
                let commitment = selection.commitment;
                anyhow::Ok((selection, commitment))
            })
            .expect("auto never fails")
    }

    /// A load made now is admitted against the first engine: it does not
    /// get the planes that engine still holds.
    fn assert_first_engine_still_counted(&self, stage: &str) {
        assert_eq!(self.ledger.total(), self.one_engine, "{stage}");
        let (selection, reservation) = self.load();
        assert_eq!(
            selection.memory.map(|memory| memory.resident_bytes),
            Some(self.one_engine),
            "{stage}"
        );
        assert_eq!(selection.commitment.plane_bytes, 0, "{stage}");
        assert!(selection.routes.any_withheld(), "{stage}");
        drop(reservation);
        // Loads racing each other while the teardown is blocked fare no
        // better: none of them gets the planes.
        std::thread::scope(|scope| {
            let loads: Vec<_> = (0..4).map(|_| scope.spawn(|| self.load())).collect();
            for load in loads {
                let (selection, _reservation) = load.join().expect("load");
                assert_eq!(selection.commitment.plane_bytes, 0, "{stage}");
            }
        });
        assert_eq!(self.ledger.total(), self.one_engine, "{stage}");
    }
}

/// Stands for something the worker owns: reports when it is dropped.
struct Owned(&'static str, Sender<&'static str>);

impl Drop for Owned {
    fn drop(&mut self) {
        let _ = self.1.send(self.0);
    }
}

thread_local! {
    /// A thread-local cache the explicit release does not know about.
    static OTHER_CACHE: std::cell::RefCell<Option<Owned>> = const { std::cell::RefCell::new(None) };
}

/// A gate a teardown step waits at.
fn gate() -> (Sender<()>, Receiver<()>) {
    channel()
}

#[test]
fn another_load_cannot_claim_an_engines_memory_until_its_teardown_is_complete() {
    let host = Host::new();
    let (first, reservation) = host.load();
    assert!(!first.routes.any_withheld(), "alone it gets every plane");
    assert_eq!(host.ledger.total(), host.one_engine);

    let (events, seen) = channel::<&'static str>();
    let (stop, stopped) = gate();
    let (open_cache_release, cache_release) = gate();
    let (open_flush, flush_gate) = gate();
    let model = Owned("model dropped", events.clone());
    let other_cache = Owned("other thread-local destroyed", events.clone());
    let (cache_events, flush_events) = (events.clone(), events.clone());
    let teardown = WorkerTeardown::new(Some(reservation), move || {
        let _ = cache_events.send("releasing thread caches");
        let _ = cache_release.recv();
        let _ = cache_events.send("thread caches released");
        Ok(Box::new(move || {
            let _ = flush_events.send("flushing");
            let _ = flush_gate.recv();
            Ok(())
        }) as DeferredFlush)
    });
    let supervisor = EngineSupervisor::new_with_monitor_start(|_| Ok(()));
    let engine =
        spawn_supervised_worker("teardown-test", supervisor.clone(), teardown, move |_| {
            let _model = model;
            OTHER_CACHE.with(|cache| *cache.borrow_mut() = Some(other_cache));
            let _ = stopped.recv();
        })
        .expect("spawn");

    // The engine is running.
    host.assert_first_engine_still_counted("running");

    // Evicted: every handle is gone and the worker is told to stop. It
    // drops the model, and its cache release blocks.
    drop(supervisor);
    drop(stop);
    assert_eq!(seen.recv(), Ok("model dropped"));
    assert_eq!(seen.recv(), Ok("releasing thread caches"));
    host.assert_first_engine_still_counted("model dropped, thread caches not released");

    // The caches are released and the thread ends, which destroys the
    // thread-local the release did not name. The flush blocks.
    drop(open_cache_release);
    assert_eq!(seen.recv(), Ok("thread caches released"));
    assert_eq!(seen.recv(), Ok("other thread-local destroyed"));
    assert_eq!(seen.recv(), Ok("flushing"));
    host.assert_first_engine_still_counted("thread gone, residency not flushed");

    // The flush completes: only now is the memory another load's to claim.
    drop(open_flush);
    engine.join().expect("teardown thread");
    assert_eq!(host.ledger.total(), 0);
    let (after, _reservation) = host.load();
    assert!(!after.routes.any_withheld());
    assert_eq!(after.commitment.total(), host.one_engine);
}

/// Runs one worker to the end of its teardown with the given outcomes and
/// returns the ledger total afterwards.
fn total_after_teardown(
    body_panics: bool,
    release: fn() -> anyhow::Result<()>,
    flush: fn() -> anyhow::Result<()>,
) -> u64 {
    let host = Host::new();
    let (_selection, reservation) = host.load();
    let (events, seen) = channel::<&'static str>();
    let model = Owned("model dropped", events);
    let teardown = WorkerTeardown::new(Some(reservation), move || {
        release()?;
        Ok(Box::new(flush) as DeferredFlush)
    });
    let supervisor = EngineSupervisor::new_with_monitor_start(|_| Ok(()));
    spawn_supervised_worker(
        "teardown-outcome-test",
        supervisor.clone(),
        teardown,
        move |_| {
            let _model = model;
            if body_panics {
                panic!("synthetic worker failure");
            }
        },
    )
    .expect("spawn")
    .join()
    .expect("teardown thread");
    assert_eq!(seen.recv(), Ok("model dropped"));
    assert_eq!(supervisor.is_healthy(), !body_panics);
    let total = host.ledger.total();
    assert!(total == 0 || total == host.one_engine);
    // A stranded engine is admitted against like a resident one.
    if total != 0 {
        host.assert_first_engine_still_counted("stranded");
    }
    total
}

fn done() -> anyhow::Result<()> {
    Ok(())
}

fn failed() -> anyhow::Result<()> {
    anyhow::bail!("synthetic Metal failure")
}

fn panicked() -> anyhow::Result<()> {
    panic!("synthetic teardown panic")
}

#[test]
fn a_clean_teardown_releases_the_memory_also_after_the_worker_panicked() {
    assert_eq!(total_after_teardown(false, done, done), 0);
    // The panic unwound the body and dropped the model; the teardown that
    // followed is complete.
    assert_eq!(total_after_teardown(true, done, done), 0);
}

#[test]
fn a_failed_teardown_keeps_the_memory_charged() {
    let one_engine = Host::new().one_engine;
    for body_panics in [false, true] {
        for (release, flush) in [
            (
                failed as fn() -> anyhow::Result<()>,
                done as fn() -> anyhow::Result<()>,
            ),
            (panicked, done),
            (done, failed),
            (done, panicked),
        ] {
            assert_eq!(
                total_after_teardown(body_panics, release, flush),
                one_engine
            );
        }
    }
}

#[test]
fn a_worker_that_never_stops_keeps_the_memory_charged() {
    let host = Host::new();
    let (_selection, reservation) = host.load();
    let (stop, stopped) = gate();
    let teardown = WorkerTeardown::new(Some(reservation), || {
        Ok(Box::new(|| Ok(())) as DeferredFlush)
    });
    let supervisor = EngineSupervisor::new_with_monitor_start(|_| Ok(()));
    let engine = spawn_supervised_worker(
        "teardown-wedged-test",
        supervisor.clone(),
        teardown,
        move |_| {
            let _ = stopped.recv();
        },
    )
    .expect("spawn");
    // Every handle is gone; the worker does not return.
    drop(supervisor);
    host.assert_first_engine_still_counted("wedged");
    assert!(!engine.is_finished());
    // Let the test's thread go.
    drop(stop);
    engine.join().expect("teardown thread");
    assert_eq!(host.ledger.total(), 0);
}

#[test]
fn a_worker_without_a_reservation_tears_down_the_same_way() {
    let (events, seen) = channel::<&'static str>();
    let flush_events = events.clone();
    let teardown = WorkerTeardown::new(None, move || {
        let _ = events.send("thread caches released");
        Ok(Box::new(move || {
            let _ = flush_events.send("flushed");
            Ok(())
        }) as DeferredFlush)
    });
    let supervisor = EngineSupervisor::new_with_monitor_start(|_| Ok(()));
    spawn_supervised_worker("teardown-unreserved-test", supervisor, teardown, |_| {})
        .expect("spawn")
        .join()
        .expect("teardown thread");
    assert_eq!(seen.recv(), Ok("thread caches released"));
    assert_eq!(seen.recv(), Ok("flushed"));
}
