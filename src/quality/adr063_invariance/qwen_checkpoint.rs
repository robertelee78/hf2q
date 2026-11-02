//! Explicit Qwen3.6 ordinary active64 evidence; no new forward/math/allocator.
use super::{protocol::Trajectory, qwen_cache, receipt::*};
use crate::inference::models::qwen35::Qwen35Config;
use anyhow::{ensure, Context, Result};

pub fn captures64(family: Family, trajectory: Trajectory) -> bool {
    family == Family::Qwen36
        && matches!(trajectory, Trajectory::T1Resume64 | Trajectory::T3Chunk64)
}
/// The one Qwen endpoint config recipe. `qwen_actual` checks it against the
/// packet identity and the endpoint metadata prepare emits its digest. `cfg`
/// is `Qwen35Config::from_gguf` (or the loaded model config) exactly as
/// parsed: no context override.
pub fn endpoint_recipe(family: Family, capacity: u32, cfg: &Qwen35Config)
    -> super::acquisition::ConfigRecipe<'static> {
    super::acquisition::ConfigRecipe {
        family,
        capacity: capacity as u64,
        slots: 1,
        output_allowance: 0,
        cache_mode: "QwenTQ8",
        route: "endpoint",
        actual_model_config_sha256: super::store::hash(format!("{cfg:?}").as_bytes()),
    }
}
pub fn expected_cache(cfg: &Qwen35Config, family: Family, trajectory: Trajectory,
    capacity: u32, mtp_positions: [usize; 2]) -> Result<Vec<ExpectedTensor>> {
    ensure!(matches!((family,cfg.variant),
        (Family::Qwen36,crate::inference::models::qwen35::Qwen35Variant::Moe) |
        (Family::Qwen38,crate::inference::models::qwen35::Qwen35Variant::Dense)),
        "explicit Qwen family/actual config variant");
    if captures64(family,trajectory) {ensure!(mtp_positions==[0,0],"Q36 ordinary target-only MTP cursors must be zero");}
    let mut out = Vec::new();
    for (index, boundary) in [128,144].into_iter().enumerate() {
        for lane in 0..4 {
            out.extend(qwen_cache::manifest(cfg,capacity,&format!("agent-{lane}"),boundary,mtp_positions[index])?);
        }
    }
    if captures64(family,trajectory) {
        for lane in 0..4 {
            // Ordinary target forwards do not execute MTP. Refuse a contrary
            // actual cursor in the caller rather than invent an active prefix.
            let before=qwen_cache::manifest(cfg,capacity,&format!("agent-{lane}"),64,0)?;
            if trajectory == Trajectory::T1Resume64 {
                for source in &before {
                    ensure!(source.at.segment<1000,"Qwen restored segment namespace");
                    let mut restored=source.clone();restored.at.segment+=1000;out.push(restored);
                }
            }
            out.extend(before);
        }
    }
    Ok(out)
}
/// Header-derived whole tensor plan BEFORE model/GPU loading. Store rechecks it.
pub fn planned_payload(cache: &[ExpectedTensor], vocab: usize, limit: u64) -> Result<u64> {
    ensure!(limit>0 && limit<=2*1024*1024*1024,"explicit finite endpoint payload limit");
    let mut bytes=576u64.checked_mul(vocab as u64).and_then(|n|n.checked_mul(4))
        .context("Qwen full144 logit byte overflow")?;
    for tensor in cache {
        ensure!(!tensor.shape.is_empty() && !tensor.shape.contains(&0),"Qwen complete expected shape");
        let n=tensor.shape.iter().try_fold(tensor.scalar.bytes() as u64,
            |n,&d|n.checked_mul(d as u64)).context("Qwen cache byte overflow")?;
        bytes=bytes.checked_add(n).context("Qwen whole payload overflow")?;
    }
    ensure!(bytes<=limit,"Qwen full logits/cache payload exceeds pre-load quota");
    Ok(bytes)
}
pub fn require_restore64_exact(tensors: &[Tensor]) -> Result<u64,String> {
    let before=tensors.iter().filter(|t|t.at.logical_step==64 && t.at.segment<1000).collect::<Vec<_>>();
    let restored=tensors.iter().filter(|t|t.at.logical_step==64 && t.at.segment>=1000).collect::<Vec<_>>();
    if before.is_empty() || before.len()!=restored.len() {return Err("Qwen fullactive64 pair missing".into());}
    let mut total=0u64;
    for source in before {
        let mut coordinate=source.at.clone();coordinate.segment+=1000;
        let matches=restored.iter().filter(|t|t.at==coordinate).collect::<Vec<_>>();
        if matches.len()!=1 {return Err("Qwen restore64 coordinate missing/duplicate".into());}
        let target=matches[0];
        if source.scalar!=target.scalar || source.shape!=target.shape || source.logical_bytes==0
            || source.logical_bytes!=target.logical_bytes || source.sha256!=target.sha256 {
            return Err("Qwen restored active64 differs in full encoded bytes".into());
        }
        total=total.checked_add(source.logical_bytes).ok_or("Qwen restore64 byte overflow")?;
    }
    Ok(total)
}
#[cfg(test)]
#[path="qwen_checkpoint_controls.rs"]
mod controls;
