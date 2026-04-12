use crate::metal::{
    instrument, BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus,
    ComputeCommandEncoder,
};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

// hf2q ADR-005 Phase 1b 1bNEW.21 vendor patch (2026-04-11):
// bump DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER from 50 to 100.
//
// Empirical sweep on HEAD `9deb18a` (Apple M5 Max, Gemma 4 26B MoE DWQ,
// canonical 187-token bench prompt, 128 decode tokens, 5-run median ×
// 3 trials):
//     CANDLE_METAL_COMPUTE_PER_BUFFER=10   → 83.5 tok/s
//     CANDLE_METAL_COMPUTE_PER_BUFFER=20   → 84.5 tok/s
//     CANDLE_METAL_COMPUTE_PER_BUFFER=50   → 85.0 tok/s (upstream default)
//     CANDLE_METAL_COMPUTE_PER_BUFFER=100  → 85.9 tok/s  ← this patch
//     CANDLE_METAL_COMPUTE_PER_BUFFER=200  → 85.9 tok/s (plateau)
//     CANDLE_METAL_COMPUTE_PER_BUFFER=5000 → 79.4 tok/s (CPU/GPU serialized)
// Net +0.9 tok/s, reproducible across 3 independent trials each,
// noise envelope ~0.3 tok/s per trial. Distributions barely overlap.
//
// Chesterton's fence cleared: the 50 value was introduced in candle
// commit 0cf516d1 (2025-09-08, "[Metal] Refactor" PR #3070) and
// preserved unchanged through db08cc0a (2025-11-11, "Add command
// buffer pool" PR #3175). Neither commit contains empirical rationale
// for 50 specifically; it is a historical carryover from the
// pre-refactor single-buffer architecture, never re-validated against
// modern Metal / modern decoder workloads. The docstring on the
// pre-refactor single-buffer struct explicitly captured the trade-off:
// "Using a single command buffer would be fastest on the GPU but
// prevents overlapping of CPU and GPU commands (because command buffer
// needs to be committed to start to work)." This is the exact
// mechanism the hf2q sweep measured — bumping to 5000 (effectively
// one buffer per forward) regressed because GPU idles until end of
// forward, while the 100 sweet spot gives the GPU enough mini-commits
// to pipeline decode-loop work against the CPU encode-ahead.
//
// Candle test safety: /opt/candle/candle-metal-kernels/src/tests.rs
// has two tests that touch this value (commands_rotation_threshold,
// commands_concurrent_acquisition); both explicitly set
// CANDLE_METAL_COMPUTE_PER_BUFFER=2 via env var at test entry, so
// neither depends on the default being 50. This patch does not break
// any candle test.
//
// Drop this override once an upstream candle release ships with a
// re-validated default at 100 or higher.
//
// See docs/spike-post-1bNEW20-results.md for the full sweep + decision
// trail and the 1bNEW.21 ADR-005 item entry for the vendor-drop plan.
const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 100;
const DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5;

/// Creates a new command buffer from the queue with an attached semaphore for tracking its state.
pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
}

/// A pool entry containing a command buffer, its usage count, and synchronization primitives.
/// The `state` mutex guards the current buffer and the in-flight list for coherent updates.
/// `compute_count` and `semaphore` remain accessible without locking for selection/coordination.
pub struct CommandBufferEntry {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

pub struct Commands {
    /// Maintains a pool of command buffers, allowing
    /// the pool to balance load across multiple buffers and improve GPU utilization.
    /// Can be shared across threads safely.
    pool: Vec<Arc<CommandBufferEntry>>,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE),
            _ => DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE,
        };

        let pool = (0..pool_size)
            .map(|_| Self::create_pool_entry(&command_queue))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            pool,
            command_queue,
            compute_per_buffer,
        })
    }

    fn create_pool_entry(
        command_queue: &CommandQueue,
    ) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(command_queue, Arc::clone(&semaphore))?;

        Ok(Arc::new(CommandBufferEntry {
            state: Mutex::new(EntryState {
                current: cb,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.compute_command_encoder())
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.blit_command_encoder())
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    // Selects an entry from the pool using a two-phase strategy:
    /// 1. Try non-blocking: find any available buffer without waiting
    /// 2. Fallback: select the least-loaded buffer and wait for availability
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Phase 1: Try to find an available buffer without blocking
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Phase 2: Select the buffer with the most work and wait for it
        let entry = self
            .pool
            .iter()
            .max_by_key(|e| e.compute_count.load(Ordering::Acquire))
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Command buffer pool is empty".to_string(),
            ))?;

        let entry = Arc::clone(entry);
        {
            let mut guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }

        Ok(entry)
    }

    /// Creates an encoder from the selected entry, recycling the buffer if needed.
    /// When recycling, the old committed buffer is moved to `in_flight` so we can later wait on it.
    fn finalize_entry<F, E>(
        &self,
        entry: Arc<CommandBufferEntry>,
        create_encoder: F,
    ) -> Result<(bool, E), MetalKernelError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        let mut state = entry.state.lock()?;

        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }

        let encoder = create_encoder(&mut state.current);

        Ok((flush, encoder))
    }

    /// Flushes all buffers and waits for their completion.
    /// Commits any pending work on the current buffers, moves them to in-flight,
    /// then waits on all in-flight buffers including those from prior recycles.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Under state lock, commit current if it has pending work and swap to a fresh one.
            let to_wait: Vec<CommandBuffer> = {
                // Ensure no active encoder is still encoding on this entry.
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));

                let mut state = entry.state.lock()?;

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(&entry, &mut state, 0)?;
                }

                // Drain `in_flight` into a local vec to wait without holding the lock.
                // Replaces `state.in_flight` with an empty vec and returns its previous contents.
                std::mem::take(&mut state.in_flight)
            };

            for cb in to_wait {
                Self::ensure_completed(&cb)?;
            }
        }

        // hf2q ADR-006 Phase 0 vendor patch (2026-04-11): resolve any
        // pending counter-sample entries now that every in-flight
        // command buffer is completed. This is the only safe point to
        // read MTLCounterSampleBuffer contents because stage-boundary
        // samples are only written after the encoder's command buffer
        // finishes executing on the GPU. `resolve_and_maybe_dump` is a
        // no-op when instrumentation is disabled.
        instrument::resolve_and_maybe_dump();

        Ok(())
    }

    /// Flushes all buffers without waiting for completion.
    /// Commits any pending work and moves current buffers to in-flight.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let _guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut state = entry.state.lock()?;

            if entry.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&entry, &mut state, 0)?;
            }
        }

        Ok(())
    }

    /// Commit the current command buffer, swap in a fresh one, push the old into `in_flight`,
    /// and reset `compute_count` to `reset_to`.
    fn commit_swap_locked(
        &self,
        entry: &CommandBufferEntry,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        state.current.commit();
        let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        entry.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // TODO: Avoid redundant allocation before drop
        let _ = self.flush();
    }
}
