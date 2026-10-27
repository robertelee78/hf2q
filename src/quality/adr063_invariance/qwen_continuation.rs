//! Explicit tail scope: ALL logical rows64..143 and active caches128/144.
//! The owning engine's first64 Last-head path is NOT full144 logits evidence.
use super::{qwen_cache, receipt::*, store::Store};
use crate::inference::models::qwen35::{kv_cache::HybridKvCache, Qwen35Config};
use anyhow::{ensure, Result};
pub const START: usize = 64;
pub const END: usize = 144;
pub fn selected_rows(start: usize, count: usize) -> Result<std::ops::Range<usize>> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| anyhow::anyhow!("tail row overflow"))?;
    ensure!(
        count > 0 && end <= END,
        "tail transaction outside fixed144-token trajectory"
    );
    Ok(start.max(START)..end.max(START))
}
fn at(agent: &str, row: usize, layers: u32) -> Coordinate {
    Coordinate {
        logical_step: 0,
        layer: layers,
        stage: Stage::Logits,
        agent: agent.into(),
        logical_row_start: row as u64,
        segment: 0,
    }
}
pub fn expected(cfg: &Qwen35Config, capacity: u32) -> Result<Vec<ExpectedTensor>> {
    let mut result = Vec::new();
    for row in START..END {
        for lane in 0..4 {
            result.push(ExpectedTensor {
                at: at(&format!("agent-{lane}"), row, cfg.num_hidden_layers),
                scalar: Scalar::F32,
                shape: vec![1, cfg.vocab_size as usize],
            });
        }
    }
    for boundary in [128, 144] {
        for lane in 0..4 {
            result.extend(qwen_cache::manifest(
                cfg,
                capacity,
                &format!("agent-{lane}"),
                boundary,
                0,
            )?);
        }
    }
    result.sort_by(|a, b| {
        let position = |e: &ExpectedTensor| {
            if e.at.stage == Stage::Logits {
                e.at.logical_row_start
            } else {
                e.at.logical_step
            }
        };
        (
            position(a),
            a.at.layer,
            a.at.stage,
            &a.at.agent,
            a.at.segment,
        )
            .cmp(&(
                position(b),
                b.at.layer,
                b.at.stage,
                &b.at.agent,
                b.at.segment,
            ))
    });
    Ok(result)
}
pub fn checkpoint_expected(cfg: &Qwen35Config, capacity: u32) -> Result<Vec<ExpectedTensor>> {
    let mut result = Vec::new();
    for lane in 0..4 {
        result.extend(qwen_cache::manifest(
            cfg,
            capacity,
            &format!("agent-{lane}"),
            64,
            0,
        )?);
    }
    Ok(result)
}
pub fn payload(expected: &[ExpectedTensor]) -> Result<u64> {
    expected.iter().try_fold(0u64, |total, e| {
        let bytes = e
            .shape
            .iter()
            .try_fold(e.scalar.bytes() as u64, |n, &d| n.checked_mul(d as u64))
            .ok_or_else(|| anyhow::anyhow!("continuation metadata payload overflow"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("continuation aggregate payload overflow"))
    })
}
pub fn after_checked_target(
    store: &mut Store,
    cfg: &Qwen35Config,
    lane: usize,
    start: usize,
    values: &[f32],
    cache: &HybridKvCache,
) -> Result<()> {
    let vocab = cfg.vocab_size as usize;
    ensure!(
        vocab > 0 && values.len() % vocab == 0,
        "target All logits extent"
    );
    let count = values.len() / vocab;
    let end = start
        .checked_add(count)
        .ok_or_else(|| anyhow::anyhow!("target position overflow"))?;
    ensure!(
        cache.sequence_len_for_slot(crate::serve::multi_seq_kv::SlotId(0))? as usize == end,
        "actual target terminal cursor"
    );
    for row in selected_rows(start, count)? {
        let local = row - start;
        let bytes = bytemuck::cast_slice(&values[local * vocab..(local + 1) * vocab]);
        store.capture(at(&format!("agent-{lane}"),row,cfg.num_hidden_layers),Scalar::F32,vec![1,vocab],values.as_ptr() as usize as u64,std::mem::size_of_val(values) as u64,(local*vocab*4) as u64,None,"actual target forward_gpu All checked return/download; declared continuation scope64..143",bytes).map_err(anyhow::Error::msg)?;
    }
    if matches!(end, 128 | 144) {
        ensure!(
            cache
                .mtp_slot
                .as_ref()
                .is_none_or(|mtp| mtp.current_len[0] == 0),
            "ordinary specOFF tail cannot omit an active MTP prefix"
        );
        qwen_cache::after_checked_terminal(
            store,
            cache,
            cfg.num_hidden_layers,
            &format!("agent-{lane}"),
            end,
        )?;
    }
    Ok(())
}
#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn q1_continuation_scope_covers_exact80_rows_without_first64_claim() {
        assert_eq!(selected_rows(0, 128).unwrap(), 64..128);
        assert_eq!(selected_rows(64, 64).unwrap(), 64..128);
        let mut ordinary: Vec<_> = selected_rows(0, 128).unwrap().collect();
        let mut recovered: Vec<_> = selected_rows(64, 64).unwrap().collect();
        for position in 128..144 {
            ordinary.extend(selected_rows(position, 1).unwrap());
            recovered.extend(selected_rows(position, 1).unwrap());
        }
        assert_eq!(ordinary, (64..144).collect::<Vec<_>>());
        assert_eq!(ordinary, recovered);
        assert_eq!(ordinary.len(), 80);
        assert!(selected_rows(144, 1).is_err());
        assert!(selected_rows(usize::MAX, 2).is_err());
        assert!(selected_rows(64, 0).is_err());
    }
    #[test]
    fn q1_continuation_actual_plan_closes_disk_quota_without_inactive_mtp() {
        for (q38, tail_bytes, snapshot_bytes, tail_records, snapshot_records, total) in [
            (
                true,
                1_609_203_712u64,
                636_092_416u64,
                3136usize,
                1408usize,
                2_881_388_544u64,
            ),
            (false, 856_074_240, 266_117_120, 1440, 560, 1_388_308_480),
        ] {
            let cfg = qwen_cache::selected_header_fixture(q38);
            let e = expected(&cfg, 262144).unwrap();
            let s = checkpoint_expected(&cfg, 262144).unwrap();
            assert_eq!(e.len(), tail_records);
            assert_eq!(s.len(), snapshot_records);
            assert_eq!(payload(&e).unwrap(), tail_bytes);
            assert_eq!(payload(&s).unwrap(), snapshot_bytes);
            assert_eq!(tail_bytes + 2 * snapshot_bytes, total);
            assert!(total < 4 * 1024 * 1024 * 1024);
            assert!(tail_bytes < 2 * 1024 * 1024 * 1024 && snapshot_bytes < 1024 * 1024 * 1024);
            assert!(e
                .iter()
                .filter(|e| e.at.stage == Stage::Logits)
                .all(|e| (64..144).contains(&e.at.logical_row_start)));
            assert!(e
                .iter()
                .filter(|e| e.at.stage != Stage::Logits)
                .all(|e| e.at.layer < cfg.num_hidden_layers));
        }
    }
}
