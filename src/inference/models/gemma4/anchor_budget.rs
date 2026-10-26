//! Requested CPU payload quota for ALL Hybrid anchor captures in the experiment.
use super::{GemmaHybridSlidingLayerAnchor, MultiSeqHybridKvBuffers};
use anyhow::{anyhow, Result};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, LazyLock,
};
pub const LIMIT: usize = 1536 * 1024 * 1024;
static QUOTA: LazyLock<Arc<Quota>> = LazyLock::new(|| {
    Arc::new(Quota {
        used: AtomicUsize::new(0),
        limit: LIMIT,
    })
});
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
pub(super) fn reserve(
    scaffold: &[MultiSeqHybridKvBuffers],
    slot: usize,
    prompt_len: usize,
) -> Result<Option<Reservation>> {
    if !crate::debug::investigation_env::gemma_full_input::selected().map_err(|e| anyhow!(e))? {
        return Ok(None);
    }
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
    acquire(&QUOTA, bytes).map(Some)
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
        assert!(crate::debug::investigation_env::gemma_full_input::selected().unwrap());
        assert_eq!(
            QUOTA.used.load(Ordering::Acquire),
            0,
            "fresh isolated process"
        );
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
        let reserved = QUOTA.used.load(Ordering::Acquire);
        assert!(reserved > anchor.total_bytes());
        let denied = snapshot_gemma_hybrid_slot_anchor(
            &layers,
            crate::serve::multi_seq_kv::SlotId(0),
            LIMIT / 4,
        )
        .err()
        .unwrap();
        assert!(denied.is::<ReservationError>());
        assert_eq!(QUOTA.used.load(Ordering::Acquire), reserved);
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
        assert_eq!(QUOTA.used.load(Ordering::Acquire), 0);
        println!("GEMMA_ANCHOR_CPU_COMPLETE roots=6 requested_metal_bytes=30880 cpu_packet_max_bytes=262144 gpu_submissions=0 restored=7_to_6 quota_released=true");
    }
}
