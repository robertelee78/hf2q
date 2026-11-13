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
//! | `engine_qwen35`: slot KV provisioning | provisioning | [`SettledDevice`] |
//! | `engine_qwen35`: the seven request functions (unary, slot-aware decode, slot-aware stream, soft tokens, vision, SerialFifo stream, embed) | KV allocation, snapshots and restores of that request | [`SettledDevice`], when the request returns or unwinds |
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
//! A settle that fails cannot be reported from a destructor, so it is
//! recorded for the thread ([`take_thread_unsettled`]); the thread's
//! teardown reads the record and fails closed: the reservation is stranded.

use std::cell::RefCell;

use anyhow::{Context, Result};
use mlx_native::MlxDevice;

/// Commit an empty command buffer on `device` and wait for it.
pub(crate) fn settle(device: &MlxDevice) -> Result<()> {
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

/// The first failure to settle a queue on this thread since the last call,
/// if there was one. A thread that owns a model asks this before the
/// model's memory is taken to be free.
pub(crate) fn take_thread_unsettled() -> Option<String> {
    UNSETTLED.with(|unsettled| unsettled.borrow_mut().take())
}

/// A device a load or a request creates for its own buffers and commands.
/// Its queue is settled when the owner drops it.
pub(crate) struct SettledDevice(MlxDevice);

impl SettledDevice {
    pub(crate) fn new() -> mlx_native::Result<Self> {
        MlxDevice::new().map(Self)
    }
}

impl std::ops::Deref for SettledDevice {
    type Target = MlxDevice;

    fn deref(&self) -> &MlxDevice {
        &self.0
    }
}

impl Drop for SettledDevice {
    fn drop(&mut self) {
        if let Err(error) = settle(&self.0) {
            let reason = format!("{error:#}");
            tracing::error!(reason, "a device's command queue could not be settled");
            UNSETTLED.with(|unsettled| {
                unsettled.borrow_mut().get_or_insert(reason);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_device_is_the_device_and_settles_when_dropped() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = SettledDevice::new() else {
            eprintln!("skipping: no Metal device");
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
        settle(&device).expect("settle");
        drop(device);
        assert!(take_thread_unsettled().is_none(), "a healthy queue settles");
    }

    #[test]
    fn an_unsettled_queue_is_recorded_once_for_the_thread() {
        std::thread::spawn(|| {
            assert!(take_thread_unsettled().is_none());
            UNSETTLED.with(|unsettled| {
                unsettled.borrow_mut().get_or_insert("first".to_owned());
                unsettled.borrow_mut().get_or_insert("second".to_owned());
            });
            assert_eq!(take_thread_unsettled().as_deref(), Some("first"));
            assert!(take_thread_unsettled().is_none());
        })
        .join()
        .expect("thread");
    }
}
