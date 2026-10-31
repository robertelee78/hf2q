//! Requested CPU payload quota for ALL Hybrid anchor captures on a Gemma
//! worker with the full-input checkpoint active. Each worker thread owns its
//! own quota, so the per-worker fit check (`QuotaPlan`) is the whole story:
//! two activated Gemma engines in one process never draw on one budget.
use super::{GemmaHybridSlidingLayerAnchor, MultiSeqHybridKvBuffers};
use anyhow::{anyhow, Result};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[cfg(test)]
#[path = "c1_anchor_budget_tests.rs"]
mod c1_anchor_budget_tests;
pub const LIMIT: usize = 1536 * 1024 * 1024;
/// Live anchor generations per slot: the published/rollback checkpoint plus
/// the new capture constructed before the old one drops. Every capture site
/// replaces (never accumulates) the slot's anchor.
pub const GENERATIONS_PER_SLOT: usize = 2;
thread_local! {
    /// The calling worker's quota. A `Reservation` keeps its `Arc`, so a
    /// payload dropped on another thread still refunds the owning worker.
    static QUOTA: Arc<Quota> = Arc::new(Quota {
        used: AtomicUsize::new(0),
        limit: LIMIT,
    });
}
fn worker_quota() -> Arc<Quota> {
    QUOTA.with(Arc::clone)
}
struct Quota {
    used: AtomicUsize,
    limit: usize,
}
#[derive(Debug, thiserror::Error)]
#[error("Gemma CPU anchor reservation refused: requested={requested} used={used} limit={limit}")]
pub struct ReservationError {
    requested: usize,
    used: usize,
    limit: usize,
}
pub struct Reservation {
    quota: Arc<Quota>,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.quota.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
fn acquire(quota: &Arc<Quota>, bytes: usize) -> Result<Reservation> {
    let mut used = quota.used.load(Ordering::Acquire);
    loop {
        let Some(next) = used.checked_add(bytes).filter(|next| *next <= quota.limit) else {
            return Err(ReservationError {
                requested: bytes,
                used,
                limit: quota.limit,
            }
            .into());
        };
        match quota
            .used
            .compare_exchange_weak(used, next, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                return Ok(Reservation {
                    quota: Arc::clone(quota),
                    bytes,
                })
            }
            Err(actual) => used = actual,
        }
    }
}
fn add(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("Gemma anchor requested-byte overflow"))?;
    Ok(())
}
/// Worst-case requested bytes for every slot at two generations, computed
/// from the actual provisioned Hybrid scaffold before serving starts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuotaPlan {
    pub per_slot_bytes: usize,
    pub slots: usize,
    pub total_bytes: Option<usize>,
    pub limit: usize,
}
impl QuotaPlan {
    pub fn new(per_slot_bytes: usize, slots: usize, limit: usize) -> Self {
        Self {
            per_slot_bytes,
            slots,
            total_bytes: per_slot_bytes
                .checked_mul(GENERATIONS_PER_SLOT)
                .and_then(|bytes| bytes.checked_mul(slots)),
            limit,
        }
    }
    /// No in-domain capture can be refused when every slot holds its maximum
    /// generations at the longest admissible prompt ledger.
    pub fn fits(&self) -> bool {
        self.slots > 0 && self.total_bytes.is_some_and(|total| total <= self.limit)
    }
}
pub fn quota_plan(
    scaffold: &[MultiSeqHybridKvBuffers],
    max_prompt_len: usize,
) -> Result<QuotaPlan> {
    let slots = scaffold
        .first()
        .map(|layer| layer.n_seqs as usize)
        .ok_or_else(|| anyhow!("Gemma anchor quota plan needs a Hybrid scaffold"))?;
    let mut per_slot_bytes = 0;
    for slot in 0..slots {
        per_slot_bytes = per_slot_bytes.max(requested_bytes(scaffold, slot, max_prompt_len)?);
    }
    Ok(QuotaPlan::new(per_slot_bytes, slots, LIMIT))
}
fn requested_bytes(
    scaffold: &[MultiSeqHybridKvBuffers],
    slot: usize,
    prompt_len: usize,
) -> Result<usize> {
    let mut bytes = prompt_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow!("Gemma anchor token-ledger byte overflow"))?;
    add(
        &mut bytes,
        scaffold
            .len()
            .checked_mul(std::mem::size_of::<Option<GemmaHybridSlidingLayerAnchor>>())
            .ok_or_else(|| anyhow!("Gemma anchor layer metadata overflow"))?,
    )?;
    for layer in scaffold {
        let n = layer.n_seqs as usize;
        anyhow::ensure!(n > 0 && slot < n, "Gemma anchor budget invalid slot/count");
        if !layer.is_sliding {
            continue;
        }
        for buffer in [&layer.k, &layer.v_packed]
            .into_iter()
            .chain((layer.v_norms.byte_len() != 4).then_some(&layer.v_norms))
            .chain(layer.bf16_xlen_k.as_ref())
            .chain(layer.bf16_xlen_v.as_ref())
        {
            anyhow::ensure!(
                buffer.byte_len() % n == 0,
                "Gemma anchor budget indivisible slot bytes"
            );
            add(&mut bytes, buffer.byte_len() / n)?;
        }
    }
    add(
        &mut bytes,
        std::mem::size_of::<super::GemmaHybridSlotAnchor>(),
    )?;
    add(
        &mut bytes,
        std::mem::size_of::<Vec<u32>>() + std::mem::size_of::<Option<[u8; 32]>>(),
    )?;
    Ok(bytes)
}
pub(super) fn reserve(
    scaffold: &[MultiSeqHybridKvBuffers],
    slot: usize,
    prompt_len: usize,
) -> Result<Option<Reservation>> {
    if !crate::debug::investigation_env::gemma_full_input::worker_active() {
        return Ok(None);
    }
    acquire(
        &worker_quota(),
        requested_bytes(scaffold, slot, prompt_len)?,
    )
    .map(Some)
}
/// Requested bytes currently reserved on this worker thread (test observability).
#[cfg(test)]
pub fn used_bytes() -> usize {
    worker_quota().used.load(Ordering::Acquire)
}
/// The exact error a refused reservation returns (for degrade-path tests).
#[cfg(test)]
pub fn refusal_for_test() -> anyhow::Error {
    let quota = Arc::new(Quota {
        used: AtomicUsize::new(0),
        limit: 0,
    });
    acquire(&quota, 1).err().expect("zero quota refuses")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemma_anchor_budget_denial_preserves_existing_reservation() {
        let q = Arc::new(Quota {
            used: AtomicUsize::new(0),
            limit: 10,
        });
        let a = acquire(&q, 7).unwrap();
        assert!(acquire(&q, 4).is_err());
        assert_eq!(q.used.load(Ordering::Acquire), 7);
        drop(a);
        assert_eq!(q.used.load(Ordering::Acquire), 0);
        assert!(acquire(&q, usize::MAX).is_err());
    }
    #[test]
    fn gemma_anchor_budget_covers_four_old_and_new_generations() {
        let q = Arc::new(Quota {
            used: AtomicUsize::new(0),
            limit: LIMIT,
        });
        let bytes = 158_105_600 + 262_144 * 4 + 30 * 256;
        let held: Vec<_> = (0..8).map(|_| acquire(&q, bytes).unwrap()).collect();
        assert_eq!(q.used.load(Ordering::Acquire), 8 * bytes);
        assert!(acquire(&q, LIMIT).is_err());
        drop(held);
        assert_eq!(q.used.load(Ordering::Acquire), 0);
    }
    #[test]
    fn gemma_anchor_budget_four_concurrent_captures_do_not_overbook() {
        let q = Arc::new(Quota {
            used: AtomicUsize::new(0),
            limit: LIMIT,
        });
        let hold = Arc::new(std::sync::Barrier::new(5));
        let release = Arc::new(std::sync::Barrier::new(5));
        let joins: Vec<_> = (0..4)
            .map(|_| {
                let (q, hold, release) = (Arc::clone(&q), Arc::clone(&hold), Arc::clone(&release));
                std::thread::spawn(move || {
                    let ticket = acquire(&q, LIMIT / 4 + 1);
                    hold.wait();
                    release.wait();
                    ticket.is_ok()
                })
            })
            .collect();
        hold.wait();
        let used = q.used.load(Ordering::Acquire);
        release.wait();
        let successes = joins
            .into_iter()
            .map(|join| join.join().unwrap())
            .filter(|ok| *ok)
            .count();
        assert_eq!(successes, 3);
        assert_eq!(used, 3 * (LIMIT / 4 + 1));
        assert_eq!(q.used.load(Ordering::Acquire), 0);
    }
    #[test]
    fn gemma_anchor_budget_payload_drops_before_refund() {
        struct Payload {
            q: Arc<Quota>,
        }
        impl Drop for Payload {
            fn drop(&mut self) {
                assert_eq!(self.q.used.load(Ordering::Acquire), 7);
            }
        }
        struct Owned {
            _payload: Payload,
            _reservation: Reservation,
        }
        let q = Arc::new(Quota {
            used: AtomicUsize::new(0),
            limit: 10,
        });
        drop(Owned {
            _payload: Payload { q: Arc::clone(&q) },
            _reservation: acquire(&q, 7).unwrap(),
        });
        assert_eq!(q.used.load(Ordering::Acquire), 0);
    }
    /// Official Google Gemma 4 26B-A4B per-slot request: sliding payload
    /// (25 sliding layers, window 1024, TQ8 packed V) + 262144-token ledger.
    const OFFICIAL_PER_SLOT: usize = 158_105_600 + 262_144 * 4 + 30 * 256;
    /// Two activated four-slot workers each hold their full in-domain plan at
    /// the same time; neither can borrow or exhaust the other's budget.
    #[test]
    fn gemma_anchor_quota_is_per_worker_so_two_full_workers_never_refuse() {
        let plan = QuotaPlan::new(OFFICIAL_PER_SLOT, 4, LIMIT);
        assert!(plan.fits());
        let held = Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let held = Arc::clone(&held);
                std::thread::spawn(move || {
                    let quota = worker_quota();
                    let full: Vec<_> = (0..GENERATIONS_PER_SLOT * plan.slots)
                        .map(|_| acquire(&quota, plan.per_slot_bytes))
                        .collect::<Result<_>>()
                        .expect("in-domain plan never refuses on its own worker");
                    held.wait();
                    let used = used_bytes();
                    let beyond = acquire(&quota, LIMIT - used + 1).is_err();
                    drop(full);
                    (used, beyond, used_bytes())
                })
            })
            .collect();
        for worker in workers {
            let (used, beyond, after) = worker.join().unwrap();
            assert_eq!(Some(used), plan.total_bytes, "only this worker's bytes");
            assert!(beyond, "past its own limit the worker refuses");
            assert_eq!(after, 0);
        }
        assert_eq!(used_bytes(), 0, "the test thread's quota was never touched");
    }
}

#[cfg(test)]
mod actual_cpu {
    use super::super::{
        alloc_multi_seq_hybrid_kv_for_layer, restore_gemma_hybrid_slot_anchor,
        snapshot_gemma_hybrid_slot_anchor,
    };
    use super::*;
    #[test]
    #[ignore = "bounded actual Shared Metal CPU-copy control; no GPU submission"]
    fn gemma_anchor_cpu_copy_reservation_and_exact_retry_layout() {
        crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
        assert_eq!(used_bytes(), 0, "fresh isolated process");
        let dev = mlx_native::MlxDevice::new().expect("actual Metal device required");
        let mut layers = vec![
            alloc_multi_seq_hybrid_kv_for_layer(&dev, 0, 1, 256, 4, true, 2).unwrap(),
            alloc_multi_seq_hybrid_kv_for_layer(&dev, 1, 1, 512, 8, false, 2).unwrap(),
        ];
        let mut initial = Vec::new();
        for (i, layer) in layers.iter_mut().enumerate() {
            assert!(layer.bf16_xlen_k.is_none() && layer.bf16_xlen_v.is_none());
            assert_eq!(layer.v_packed.dtype(), mlx_native::DType::U8);
            let mut copies = Vec::new();
            for (role, b) in [&mut layer.k, &mut layer.v_packed, &mut layer.v_norms]
                .into_iter()
                .enumerate()
            {
                let data = b.as_mut_slice::<u8>().unwrap();
                for (j, byte) in data.iter_mut().enumerate() {
                    *byte = ((j / 251 + j % 251 + i * 19 + role * 43) % 251) as u8;
                }
                copies.push(data.to_vec());
            }
            layer.seq_lens.fill(7);
            initial.push(copies);
        }
        let anchor =
            snapshot_gemma_hybrid_slot_anchor(&layers, crate::serve::multi_seq_kv::SlotId(0), 7)
                .unwrap();
        let reserved = used_bytes();
        assert!(reserved > anchor.total_bytes());
        let denied = snapshot_gemma_hybrid_slot_anchor(
            &layers,
            crate::serve::multi_seq_kv::SlotId(0),
            LIMIT / 4,
        )
        .err()
        .unwrap();
        assert!(denied.is::<ReservationError>());
        assert_eq!(used_bytes(), reserved);
        // CPU mutation simulates ring overwrite; this is not a kernel proof.
        let sliding = &mut layers[0];
        for b in [&mut sliding.k, &mut sliding.v_packed, &mut sliding.v_norms] {
            let n = b.byte_len() / 2;
            b.as_mut_slice::<u8>().unwrap()[..n].fill(253);
        }
        restore_gemma_hybrid_slot_anchor(
            &mut layers,
            crate::serve::multi_seq_kv::SlotId(0),
            &anchor,
            6,
        )
        .unwrap();
        for (i, layer) in layers.iter().enumerate() {
            for (role, b) in [&layer.k, &layer.v_packed, &layer.v_norms]
                .into_iter()
                .enumerate()
            {
                assert_eq!(b.as_slice::<u8>().unwrap(), initial[i][role]);
            }
            assert_eq!(layer.seq_lens, [6, 7]);
        }
        drop(anchor);
        assert_eq!(used_bytes(), 0);
        println!("GEMMA_ANCHOR_CPU_COMPLETE roots=6 requested_metal_bytes=30880 cpu_packet_max_bytes=262144 gpu_submissions=0 restored=7_to_6 quota_released=true");
    }
}
