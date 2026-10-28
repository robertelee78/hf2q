//! Explicit committed versus rejected-tail scopes for real Qwen K3 diagnostics.
use super::{qwen_cache, qwen_continuation::payload, receipt::*};
use crate::inference::models::qwen35::Qwen35Config;
use anyhow::{ensure, Result};
pub const CHECKPOINTS: [usize; 3] = [129, 131, 144];
pub fn at(lane: usize, position: usize, layer: u32, stage: Stage) -> Coordinate {
    Coordinate {
        logical_step: if stage == Stage::Logits {
            0
        } else {
            position as u64
        },
        layer,
        stage,
        agent: format!("agent-{lane}"),
        logical_row_start: if stage == Stage::Logits {
            position as u64
        } else {
            0
        },
        segment: 0,
    }
}
pub fn committed(cfg: &Qwen35Config, capacity: u32) -> Result<Vec<ExpectedTensor>> {
    let mut result = Vec::new();
    for position in 128..144 {
        for lane in 0..4 {
            result.push(ExpectedTensor {
                at: at(lane, position, cfg.num_hidden_layers, Stage::Logits),
                scalar: Scalar::F32,
                shape: vec![1, cfg.vocab_size as usize],
            });
        }
    }
    for position in CHECKPOINTS {
        for lane in 0..4 {
            result.extend(qwen_cache::manifest(
                cfg,
                capacity,
                &format!("agent-{lane}"),
                position,
                position,
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
pub fn raw(cfg: &Qwen35Config) -> Vec<ExpectedTensor> {
    let mut result = Vec::new();
    for lane in 0..4 {
        for (stage, scalar, shape) in [
            (
                Stage::SpecVerifierLogits,
                Scalar::F32,
                vec![4, cfg.vocab_size as usize],
            ),
            (
                Stage::SpecVerifierHidden,
                Scalar::F32,
                vec![4, cfg.hidden_size as usize],
            ),
            (Stage::SpecOriginalDraftIds, Scalar::U32, vec![3]),
            (Stage::SpecForcedDraftIds, Scalar::U32, vec![3]),
            (Stage::SpecCanonicalIds, Scalar::U32, vec![2]),
        ] {
            result.push(ExpectedTensor {
                at: at(lane, 129, cfg.num_hidden_layers, stage),
                scalar,
                shape,
            });
        }
    }
    result
}
pub fn paired_payload(cfg: &Qwen35Config, capacity: u32) -> Result<(u64, u64, u64)> {
    let committed = payload(&committed(cfg, capacity)?)?;
    let raw = payload(&raw(cfg))?;
    let total = committed
        .checked_mul(2)
        .and_then(|n| n.checked_add(raw))
        .ok_or_else(|| anyhow::anyhow!("T5 paired payload overflow"))?;
    ensure!(
        total <= 4 * 1024 * 1024 * 1024,
        "T5 paired tensor payload exceeds4GiB, no truncation"
    );
    Ok((committed, raw, total))
}
pub fn bad_proposal(expected: u32, vocab: u32, eos: &[u32]) -> Result<u32> {
    ensure!(expected < vocab, "canonical ID outside vocabulary");
    (0..vocab)
        .find(|id| *id != expected && !eos.contains(id))
        .ok_or_else(|| anyhow::anyhow!("no legal frozen rejection proposal"))
}
#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn q1_t5_actual_manifest_pair_fits4g_without_fabricated_decisions() {
        for (q38, base, raw, total) in [
            (true, 2_003_429_888, 16_220_288, 4_023_080_064u64),
            (false, 872_421_120, 16_023_680, 1_760_865_920),
        ] {
            let cfg = qwen_cache::selected_header_fixture(q38);
            assert_eq!(paired_payload(&cfg, 262144).unwrap(), (base, raw, total));
            assert!(committed(&cfg, 262144)
                .unwrap()
                .iter()
                .filter(|e| e.at.stage != Stage::Logits)
                .all(|e| CHECKPOINTS.contains(&(e.at.logical_step as usize))));
            assert!(super::raw(&cfg)
                .iter()
                .filter(|e| e.at.stage == Stage::SpecCanonicalIds)
                .all(|e| e.shape == [2]));
        }
    }
    #[test]
    fn q1_t5_bad_id_is_valid_nonterminal_and_frozen_not_target_mutation() {
        assert_eq!(bad_proposal(0, 4, &[1, 2]).unwrap(), 3);
        assert!(bad_proposal(0, 2, &[1]).is_err());
        assert!(bad_proposal(4, 4, &[]).is_err());
    }
}
