//! Settling the command queue of a Metal device.
//!
//! Two things outlive the handles a thread drops. A dropped buffer only
//! stages its removal from its device's residency set, and the set keeps
//! the allocation until its next commit, which a thread that encodes nothing
//! further never issues. And a command buffer that was committed without a
//! wait (the forward pipeline commits most of its command buffers that way
//! and waits at the next read) keeps every buffer it references until it
//! has completed. [`settle`] ends both for one device: it commits an empty
//! command buffer, which commits the staged removals, and waits for it,
//! which, a queue executing its command buffers in commit order, is after
//! every command committed earlier on that queue.
//!
//! An engine's memory reservation is released only when every queue its
//! model's threads encoded on is settled (`serve::api::engine`,
//! `model_worker_teardown`). A queue belongs to an `MlxDevice`, and the
//! threads of a text engine get their devices in these places and no other
//! (every `MlxDevice::new()` outside test code, 2026-11-13):
//!
//! | Where the device is created | Who encodes on it | Settled by |
//! |---|---|---|
//! | `serve::gpu::GpuContext` (the model owns it: Gemma, DeepSeek-V4, Qwen3-VL) | the forward of those families | the worker's teardown |
//! | `qwen35::forward_gpu`, the thread's GPU weight cache (two entry points, one cache) | the Qwen forward, the drafter, every `with_gpu_cache_mut` user | the worker's teardown; on the loading thread, `serve::load_engine_on_route` |
//! | `qwen35::model::Qwen35Model::load_from_gguf`; `engine_qwen35`: GLP bind, DWQ overlay | nobody: the load creates buffers on them and commits no command | must not be (below) |
//! | `engine_qwen35`: slot KV provisioning | provisioning | [`SettledDevice`], before provisioning reports success |
//! | `engine_qwen35`: the five SerialFifo request functions (unary, soft tokens, vision, stream, embed) | KV allocation, snapshots and restores of that request | `RequestDevice` (`engine_qwen35`), under the engine's supervision, before the request reports success |
//! | `engine_qwen35` and `engine_qwen3vl` load info (`backend_chip`) | nobody: only the device's name is read | not needed |
//!
//! The weight-loading device must stay uncommitted. Its buffers map the GGUF
//! file; one commit on that device makes them committed members of its
//! residency set, and they then stay resident after every handle is gone
//! (observed: with a settle added at the end of `load_from_gguf`, 20.4 GiB
//! of a torn-down Qwen3.6 engine never came back; without it, 0.18 GiB
//! were left). The same observation shows that nothing encodes on it: any
//! command would have had that effect.
//!
//! The other creation sites belong to no text engine's reservation: the
//! vision tower and its projector, the embedding models, the host probe of
//! `dflash2_policy`, the CLI commands and startup steps of `serve::mod`, the
//! request handler that builds vision soft tokens, and the calibration,
//! quantization and quality tools.
//!
//! A step that succeeds settles its queue explicitly and reports a failure
//! as its own error, before it reports success. A step that fails or
//! unwinds leaves the settle to a destructor, which cannot report. In both
//! cases a settle that fails is recorded for the thread
//! ([`take_thread_unsettled`]), and whoever owns the model on that thread
//! reads the record and fails closed: the reservation is stranded.

use std::cell::RefCell;

use anyhow::{Context, Result};
use mlx_native::MlxDevice;

/// Commit an empty command buffer on `device` and wait for it.
pub(crate) fn settle(device: &MlxDevice) -> Result<()> {
    #[cfg(test)]
    if let Some(outcome) = fault::next() {
        return outcome;
    }
    device
        .command_encoder()
        .context("create the encoder that settles the device")?
        .commit_and_wait()
        .context("commit the staged residency removals and wait for the queue")?;
    Ok(())
}

thread_local! {
    /// Why a queue this thread encoded on could not be settled.
    static UNSETTLED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Record that a queue this thread encoded on could not be settled. The
/// first reason is kept.
pub(crate) fn record_thread_unsettled(reason: impl Into<String>) {
    UNSETTLED.with(|unsettled| {
        unsettled.borrow_mut().get_or_insert_with(|| reason.into());
    });
}

/// The first failure to settle a queue on this thread since the last call,
/// if there was one. A thread that owns a model asks this before the
/// model's memory is taken to be free.
pub(crate) fn take_thread_unsettled() -> Option<String> {
    UNSETTLED.with(|unsettled| unsettled.borrow_mut().take())
}

/// [`settle`], with a failure also recorded for the thread. For a caller
/// whose own error path may be overtaken by another error before it is
/// reported: the record survives that, and the thread's owner fails closed
/// on it.
pub(crate) fn settle_or_record(device: &MlxDevice) -> Result<()> {
    let settled = settle(device);
    if let Err(error) = &settled {
        tracing::error!(
            reason = %format!("{error:#}"),
            "a device's command queue could not be settled"
        );
        record_thread_unsettled(format!("{error:#}"));
    }
    settled
}

/// A device a load step creates for its own buffers and commands, with a
/// queue that is settled before the step is done.
///
/// A step that succeeds settles it itself, with [`Self::settle`], and
/// reports a failure as its own. A step that fails or unwinds leaves it to
/// the destructor, which cannot report: it records a failure for the thread.
/// Either way a queue that could not be settled is on the thread's record.
pub(crate) struct SettledDevice {
    device: MlxDevice,
    settled: bool,
}

impl SettledDevice {
    pub(crate) fn new() -> mlx_native::Result<Self> {
        MlxDevice::new().map(|device| Self {
            device,
            settled: false,
        })
    }

    /// Settle the queue now. It is not settled a second time when this
    /// returns an error.
    pub(crate) fn settle(mut self) -> Result<()> {
        self.settled = true;
        settle_or_record(&self.device)
    }
}

impl std::ops::Deref for SettledDevice {
    type Target = MlxDevice;

    fn deref(&self) -> &MlxDevice {
        &self.device
    }
}

impl Drop for SettledDevice {
    fn drop(&mut self) {
        if !self.settled {
            // The owner is failing or unwinding: its error is already on its
            // way, and this one goes on the thread's record.
            let _ = settle_or_record(&self.device);
        }
    }
}

/// Makes [`settle`] fail or block on the calling thread, for the tests of
/// what a settle that goes wrong does to a request, a load and a teardown.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::RefCell;

    type Fault = Box<dyn FnMut() -> anyhow::Result<()>>;

    thread_local! {
        static FAULT: RefCell<Option<Fault>> = const { RefCell::new(None) };
    }

    /// Every [`super::settle`] on this thread returns what `fault` returns,
    /// without touching the device, until [`clear`].
    pub(crate) fn inject(fault: impl FnMut() -> anyhow::Result<()> + 'static) {
        FAULT.with(|slot| *slot.borrow_mut() = Some(Box::new(fault)));
    }

    pub(crate) fn clear() {
        FAULT.with(|slot| slot.borrow_mut().take());
    }

    pub(super) fn next() -> Option<anyhow::Result<()>> {
        // Taken out for the call so that a fault may itself settle or panic.
        let mut fault = FAULT.with(|slot| slot.borrow_mut().take())?;
        let outcome = fault();
        FAULT.with(|slot| {
            slot.borrow_mut().get_or_insert(fault);
        });
        Some(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

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

    fn device_or_skip() -> Option<SettledDevice> {
        let device = SettledDevice::new().ok();
        if device.is_none() {
            eprintln!("skipping: no Metal device");
        }
        device
    }

    #[test]
    fn a_settled_device_is_the_device_and_settles_a_healthy_queue() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = device_or_skip() else {
            return;
        };
        assert!(take_thread_unsettled().is_none());
        let buffer = device
            .alloc_buffer(1024, mlx_native::DType::F32, vec![256])
            .expect("buffer");
        // A command committed without a wait still references the buffer.
        let mut encoder = device.command_encoder().expect("encoder");
        encoder.commit();
        drop(buffer);
        device.settle().expect("a healthy queue settles");
        // The failing owner's path: the destructor settles.
        drop(device_or_skip());
        assert!(take_thread_unsettled().is_none());
    }

    #[test]
    fn a_step_that_succeeds_reports_a_settle_failure_as_its_own() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = device_or_skip() else {
            return;
        };
        let calls = failing_settle();
        let error = device.settle().expect_err("the step fails");
        assert!(format!("{error:#}").contains("synthetic Metal failure"));
        // Asked once: the destructor did not try again.
        assert_eq!(calls.get(), 1);
        // And it is on the thread's record for whoever owns the model.
        assert!(take_thread_unsettled()
            .expect("recorded")
            .contains("synthetic Metal failure"));
        fault::clear();
    }

    #[test]
    fn a_step_that_fails_or_unwinds_leaves_the_failure_on_the_threads_record() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let (Some(failed), Some(unwound)) = (device_or_skip(), device_or_skip()) else {
            return;
        };
        let calls = failing_settle();
        // An error return drops the device unsettled.
        let step = |device: SettledDevice| -> Result<()> {
            let _device = device;
            anyhow::bail!("the step's own error")
        };
        assert!(step(failed).is_err());
        assert_eq!(calls.get(), 1);
        assert!(take_thread_unsettled().is_some());
        // So does a panic.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _device = unwound;
            panic!("the step panicked");
        }));
        assert!(panicked.is_err());
        assert_eq!(calls.get(), 2);
        assert!(take_thread_unsettled().is_some());
        fault::clear();
    }

    #[test]
    fn a_recorded_failure_survives_the_error_that_overtakes_it() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Some(device) = device_or_skip() else {
            return;
        };
        let calls = failing_settle();
        // What `provision_dflash2` does: release, then report the earlier
        // error first.
        let provision = || -> Result<()> {
            let runtime: Result<()> = Err(anyhow::anyhow!("the drafter did not load"));
            let released = settle_or_record(&device);
            runtime?;
            released
        };
        let error = provision().expect_err("fails");
        assert!(format!("{error:#}").contains("the drafter did not load"));
        assert_eq!(calls.get(), 1);
        assert!(take_thread_unsettled()
            .expect("the cleanup failure is not lost")
            .contains("synthetic Metal failure"));
        fault::clear();
        device.settle().expect("settles");
    }

    #[test]
    fn an_unsettled_queue_is_recorded_once_for_the_thread() {
        std::thread::spawn(|| {
            assert!(take_thread_unsettled().is_none());
            record_thread_unsettled("first");
            record_thread_unsettled("second");
            assert_eq!(take_thread_unsettled().as_deref(), Some("first"));
            assert!(take_thread_unsettled().is_none());
        })
        .join()
        .expect("thread");
    }
}
