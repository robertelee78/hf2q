//! Gemma 4 batched-prefill activation arena (ADR-063 Q4-GEMMA host P1).
//!
//! `forward_prefill_batched_inner` used to `alloc_buffer` about 45 fresh
//! `[rows, …]` activations per forward. Every one paid a CPU memset of its
//! bytes plus a residency-set add/remove: about 0.02 ms per stream row, or
//! 15–42 ms of host time ahead of every warm joint forward. This arena keeps
//! those activations across forwards with the DeepSeek-V4 transient pattern
//! (`deepseek4/forward_support.rs`, `PREFILL_POOL`): one thread-local
//! [`MlxBufferPool`] per engine worker, `alloc_uninitialized` handouts and an
//! arena reset when the cycle ends. Recycled buffers stay in the pool's
//! residency set.
//!
//! Each handout states its initialization contract:
//!
//! * [`Init::Overwritten`] — every byte any kernel or host read can observe is
//!   written earlier in the same forward. The handout keeps whatever bytes the
//!   buffer held, so it costs no memset.
//! * [`Init::Zeroed`] — the buffer may be read before (or without) being
//!   written. The handout zero-fills its logical bytes, which is exactly the
//!   state `MlxDevice::alloc_buffer` gave it.
//!
//! A cycle recycles its handouts only once the forward has declared its GPU
//! work complete ([`PrefillPoolCycle::gpu_quiescent`]). A cycle that unwinds
//! without that declaration (an error between async commits) keeps its
//! buffers out of circulation until the next cycle has drained the command
//! queue.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;

use anyhow::{Context, Result};
use mlx_native::{DType, MlxBuffer, MlxBufferPool, MlxDevice};

/// Upper bound on the power-of-two bucket bytes the arena holds — handed out
/// and free together — at every moment, during a forward and between
/// forwards. A warm four-lane forward of 1,420 rows takes about 0.9 GiB of
/// buckets and the joint cold forward of 1,832 rows about 1.4 GiB. A request
/// that would grow the arena past the bound first releases the free buckets
/// (earlier forwards' buffers this forward has not reused); one that still
/// does not fit falls back to a fresh zero-filled `alloc_buffer`, the
/// pre-arena behavior.
const POOL_LIMIT_BYTES: usize = 2 << 30;

/// Initialization contract of one pooled activation (module docs).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Init {
    Overwritten,
    Zeroed,
}

struct Arena {
    pool: MlxBufferPool,
    limit: usize,
    /// Free buckets by size, mirroring the pool's free list so a request
    /// knows whether it reuses a bucket or grows the arena.
    free: HashMap<usize, usize>,
    /// Buckets handed out by the active (or unwound) cycle.
    in_use: Vec<usize>,
    /// Sum of `in_use`.
    issued: usize,
    active: bool,
    /// The previous cycle ended without declaring its GPU work complete.
    undrained: bool,
}

impl Arena {
    fn new(limit: usize) -> Self {
        Self {
            pool: MlxBufferPool::new(),
            limit,
            free: HashMap::new(),
            in_use: Vec::new(),
            issued: 0,
            active: false,
            undrained: false,
        }
    }

    /// Bucket bytes the arena holds, handed out and free.
    fn held_bytes(&self) -> usize {
        self.pool.free_bytes().saturating_add(self.issued)
    }

    /// Take `bucket` for the active cycle: `false` when it fits neither a
    /// free bucket nor the limit, even after releasing every free bucket.
    fn reserve(&mut self, bucket: usize) -> bool {
        match self.free.get_mut(&bucket) {
            Some(count) if *count > 0 => *count -= 1,
            _ => {
                if self.issued.saturating_add(bucket) > self.limit {
                    return false;
                }
                if self.held_bytes().saturating_add(bucket) > self.limit {
                    self.pool.clear();
                    self.free.clear();
                }
            }
        }
        self.in_use.push(bucket);
        self.issued += bucket;
        true
    }

    /// Return the last cycle's handouts to the free list.
    fn recycle(&mut self) {
        self.pool.reset();
        for bucket in self.in_use.drain(..) {
            *self.free.entry(bucket).or_default() += 1;
        }
        self.issued = 0;
        self.undrained = false;
    }
}

thread_local! {
    static ARENA: RefCell<Arena> = RefCell::new(Arena::new(POOL_LIMIT_BYTES));
}

/// One forward's use of the arena. Drop it after every handout is dropped:
/// declare it before the first activation so it is dropped after them.
pub(crate) struct PrefillPoolCycle {
    zero_all: bool,
    quiescent: Cell<bool>,
    /// The arena is thread-local; the cycle must end on its own thread.
    _thread_bound: PhantomData<*const ()>,
}

impl PrefillPoolCycle {
    /// Open a cycle. `zero_all` turns every handout into [`Init::Zeroed`]
    /// (diagnostics that CPU-read whole buffers). `drain` must wait for every
    /// command buffer already committed to the queue; it runs only when the
    /// previous cycle unwound with GPU work possibly in flight.
    pub(crate) fn begin(zero_all: bool, drain: impl FnOnce() -> Result<()>) -> Result<Self> {
        let undrained = ARENA.with(|arena| {
            let arena = arena.borrow();
            anyhow::ensure!(!arena.active, "nested Gemma batched-prefill pool cycle");
            Ok(arena.undrained)
        })?;
        if undrained {
            drain().context("drain the queue before recycling unwound Gemma prefill buffers")?;
        }
        ARENA.with(|arena| {
            let mut arena = arena.borrow_mut();
            if arena.undrained {
                arena.recycle();
            }
            arena.active = true;
        });
        Ok(Self {
            zero_all,
            quiescent: Cell::new(false),
            _thread_bound: PhantomData,
        })
    }

    /// Hand out a `[elements]` activation under `init`'s contract.
    pub(crate) fn alloc(
        &self,
        device: &MlxDevice,
        dtype: DType,
        elements: usize,
        init: Init,
        label: &str,
    ) -> Result<MlxBuffer> {
        let bytes = elements
            .checked_mul(dtype.size_of())
            .with_context(|| format!("batched alloc {label}: byte size overflow"))?;
        let bucket = bytes.checked_next_power_of_two().unwrap_or(usize::MAX);
        let fill = self.fill_byte(init);
        ARENA.with(|arena| {
            let mut arena = arena.borrow_mut();
            let mut buffer = if bytes == 0 || !arena.reserve(bucket) {
                let buffer = device
                    .alloc_buffer(bytes, dtype, vec![elements])
                    .map_err(|e| anyhow::anyhow!("batched alloc {label}: {e}"))?;
                // `alloc_buffer` already zero-filled it.
                if fill.is_none_or(|byte| byte == 0) {
                    return Ok(buffer);
                }
                buffer
            } else {
                match arena
                    .pool
                    .alloc_uninitialized(device, bytes, dtype, vec![elements])
                {
                    Ok(buffer) => buffer,
                    Err(e) => {
                        // Undo the reservation; the pool's free list may have
                        // lost the bucket, so drop the free side entirely.
                        arena.in_use.pop();
                        arena.issued -= bucket;
                        arena.pool.clear();
                        arena.free.clear();
                        anyhow::bail!("batched pooled alloc {label}: {e}");
                    }
                }
            };
            if let Some(byte) = fill {
                buffer
                    .as_mut_slice::<u8>()
                    .map_err(|e| anyhow::anyhow!("batched alloc {label} fill: {e}"))?
                    .fill(byte);
            }
            Ok(buffer)
        })
    }

    /// The byte a handout is filled with, if any.
    fn fill_byte(&self, init: Init) -> Option<u8> {
        if self.zero_all || init == Init::Zeroed {
            return Some(0);
        }
        #[cfg(test)]
        if let Some(byte) = TEST_OVERWRITTEN_FILL.with(Cell::get) {
            return Some(byte);
        }
        None
    }

    /// Declare that no committed command buffer still uses this cycle's
    /// handouts, so dropping the cycle may recycle them.
    pub(crate) fn gpu_quiescent(&self) {
        self.quiescent.set(true);
    }
}

impl Drop for PrefillPoolCycle {
    fn drop(&mut self) {
        let quiescent = self.quiescent.get();
        ARENA.with(|arena| {
            let mut arena = arena.borrow_mut();
            arena.active = false;
            if quiescent {
                arena.recycle();
            } else {
                arena.undrained = true;
            }
        });
    }
}

#[cfg(test)]
fn set_limit_for_test(limit: usize) {
    ARENA.with(|arena| arena.borrow_mut().limit = limit);
}

/// Bucket bytes the arena holds on this thread (handed out plus free), as
/// counted by the pool itself and by the arena's mirror.
#[cfg(test)]
fn held_bytes_for_test() -> (usize, usize) {
    ARENA.with(|arena| {
        let arena = arena.borrow();
        let mirrored: usize = arena.free.iter().map(|(bucket, n)| bucket * n).sum();
        (arena.held_bytes(), mirrored + arena.issued)
    })
}

#[cfg(test)]
thread_local! {
    static TEST_OVERWRITTEN_FILL: Cell<Option<u8>> = const { Cell::new(None) };
}

/// Fill every [`Init::Overwritten`] handout on this thread with `byte`
/// (`None`: production behavior). `Some(0)` reproduces the pre-arena zeroed
/// allocations; a hostile byte such as `0xFF` (NaN in every float dtype)
/// proves on a real model that no kernel reads an `Overwritten` byte its
/// producer did not write.
#[cfg(test)]
pub(crate) fn set_overwritten_fill_for_test(byte: Option<u8>) {
    TEST_OVERWRITTEN_FILL.with(|fill| fill.set(byte));
}

#[cfg(test)]
#[path = "gemma_prefill_pool_tests.rs"]
mod tests;
