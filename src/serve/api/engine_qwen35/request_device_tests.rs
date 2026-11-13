//! What a request's settle does when it goes wrong ([`RequestDevice`]): it is
//! the request's own failure before any success is reported, it runs under
//! the engine's supervision, it is attempted once, and it leaves the worker
//! thread marked so that the model's memory is not released.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::mpsc::channel;

use super::RequestDevice;
use crate::inference::device_settle::{fault, take_thread_unsettled};
use crate::serve::api::engine_supervisor::EngineSupervisor;

/// Injects a settle that fails and returns how often it was asked.
fn failing_settle() -> Rc<Cell<u32>> {
    let calls = Rc::new(Cell::new(0));
    let counted = Rc::clone(&calls);
    fault::inject(move || {
        counted.set(counted.get() + 1);
        anyhow::bail!("synthetic Metal failure")
    });
    calls
}

/// The shape of a request function: GPU work, then the settle, then the
/// result. Returns what the client would be told.
fn request(
    supervisor: &EngineSupervisor,
    work: anyhow::Result<()>,
) -> anyhow::Result<&'static str> {
    let device = RequestDevice::new(supervisor)?;
    work?;
    device.settle()?;
    Ok("Done")
}

#[test]
fn a_request_reports_success_only_after_its_queue_is_settled() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if mlx_native::MlxDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let supervisor = EngineSupervisor::new();
    // A healthy queue: the request succeeds and nothing is recorded.
    assert_eq!(request(&supervisor, Ok(())).expect("request"), "Done");
    assert!(take_thread_unsettled().is_none());

    // A settle that fails is the request's failure: no success is reported.
    let calls = failing_settle();
    let error = request(&supervisor, Ok(())).expect_err("the request fails");
    assert!(
        format!("{error:#}").contains("synthetic Metal failure"),
        "{error:#}"
    );
    assert_eq!(
        calls.get(),
        1,
        "attempted once, not again by the destructor"
    );
    assert!(
        supervisor.is_healthy(),
        "an error is not a wedge: the engine keeps serving"
    );
    // The worker thread is marked: its teardown will keep the memory
    // reserved.
    assert!(take_thread_unsettled()
        .expect("recorded")
        .contains("synthetic Metal failure"));
    fault::clear();
}

#[test]
fn a_request_that_fails_still_settles_its_queue_once_under_supervision() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if mlx_native::MlxDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let supervisor = EngineSupervisor::new();
    // The request's own error wins; its queue is settled by the destructor.
    let error = request(&supervisor, Err(anyhow::anyhow!("the forward failed")))
        .expect_err("the request fails");
    assert_eq!(format!("{error:#}"), "the forward failed");
    assert!(take_thread_unsettled().is_none(), "a healthy queue settled");

    // With a settle that fails, the request still reports its own error,
    // and the thread is marked.
    let calls = failing_settle();
    let error = request(&supervisor, Err(anyhow::anyhow!("the forward failed")))
        .expect_err("the request fails");
    assert_eq!(format!("{error:#}"), "the forward failed");
    assert_eq!(calls.get(), 1);
    assert!(take_thread_unsettled().is_some());

    // A request that panics settles the same way.
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _device = RequestDevice::new(&supervisor).expect("device");
        panic!("the request panicked");
    }));
    assert!(panicked.is_err());
    assert_eq!(calls.get(), 2);
    assert!(take_thread_unsettled().is_some());
    fault::clear();
}

#[test]
fn a_settle_that_never_returns_is_a_supervised_transaction() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if mlx_native::MlxDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let supervisor = EngineSupervisor::new();
    let (entered, in_settle) = channel::<()>();
    let (release, released) = channel::<()>();
    let worker_supervisor = supervisor.clone();
    let worker = std::thread::spawn(move || {
        fault::inject(move || {
            let _ = entered.send(());
            let _ = released.recv();
            Ok(())
        });
        let outcome = request(&worker_supervisor, Ok(()));
        (
            outcome.map_err(|error| format!("{error:#}")),
            take_thread_unsettled(),
        )
    });
    // The request is inside its settle, and a transaction is armed for it:
    // when its deadline passes the engine is failed, so the client waiting
    // for this request is answered instead of waiting on a healthy-looking
    // engine for good.
    in_settle.recv().expect("the settle started");
    assert!(supervisor.is_healthy());
    supervisor.force_expire_for_test();
    assert!(!supervisor.is_healthy());
    assert!(
        supervisor
            .unhealthy_message()
            .contains("qwen35_request_settle"),
        "{}",
        supervisor.unhealthy_message()
    );
    // Should the settle return after all, the request still does not
    // succeed, and the thread is marked.
    drop(release);
    let (outcome, unsettled) = worker.join().expect("worker");
    assert!(outcome.is_err(), "{outcome:?}");
    assert!(unsettled.is_some());
}

#[test]
fn a_failed_request_on_an_unhealthy_engine_does_not_touch_the_gpu_again() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if mlx_native::MlxDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let supervisor = EngineSupervisor::new();
    let lease = supervisor
        .arm("synthetic-gpu", std::time::Duration::from_secs(60))
        .expect("arm");
    supervisor.force_expire_for_test();
    assert!(lease.finish().is_err());
    assert!(!supervisor.is_healthy());

    let calls = failing_settle();
    let error = request(&supervisor, Err(anyhow::anyhow!("the forward failed")))
        .expect_err("the request fails");
    assert_eq!(format!("{error:#}"), "the forward failed");
    // No unsupervised settle was attempted; the queue counts as unsettled.
    assert_eq!(calls.get(), 0);
    assert!(take_thread_unsettled().is_some());
    fault::clear();
}
