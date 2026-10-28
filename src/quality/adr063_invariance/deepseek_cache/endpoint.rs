//! Complete active selected DS state after existing terminal and publication.
//! Shares native plan arithmetic, never reads capacity tails/shared parent twice.
use crate::adr063_invariance::{capture,receipt::*,store::Store};
use crate::inference::models::deepseek4::{Deepseek4Config,cache::{Deepseek4Cache,Deepseek4CachePlan,CacheBufferPlan}};
use mlx_native::{DType,MlxBuffer};
fn scalar(dtype:DType)->Result<Scalar,String>{match dtype {DType::BF16=>Ok(Scalar::Bf16),
    DType::F32=>Ok(Scalar::F32),_=>Err("DS cache diagnostic dtype unsupported".into())}}
pub fn at(lane:usize,boundary:usize,layer:usize,stage:Stage,restored:bool)->Coordinate {
    Coordinate {logical_step:boundary as u64,layer:layer as u32,stage,agent:format!("agent-{lane}"),
        logical_row_start:0,segment:u32::from(restored)}
}
fn valid_shape(plan:&CacheBufferPlan,valid:Option<usize>)->Result<Vec<usize>,String> {
    let mut shape=plan.shape.clone();
    if let Some(rows)=valid {
        if rows==0 || shape.is_empty() || rows>shape[0] {return Err("DS active row extent invalid".into());}
        shape[0]=rows;
    }Ok(shape)
}
/// Same accepted native plan; position is an actual published token boundary.
fn roles<'a>(p:&'a crate::inference::models::deepseek4::cache::LayerCachePlan,pos:usize)
    ->Vec<(Stage,&'a CacheBufferPlan,Option<usize>)> {
    let mut out=vec![(Stage::CacheWindow,&p.window_kv,Some(pos.min(p.window_kv.shape[0])))];
    let compressed=if p.compress_ratio==0 {0}else{pos/p.compress_ratio as usize};
    if compressed>0 {if let Some(b)=p.compressed_kv.as_ref(){out.push((Stage::CacheCompressed,b,Some(compressed)));}}
    if p.compress_ratio==4 && pos/4>0 {if let Some(b)=p.indexer_kv.as_ref(){out.push((Stage::CacheIndexer,b,Some(pos/4)));}}
    for (stage,b) in [(Stage::CacheMainKv,p.main_kv_state.as_ref()),(Stage::CacheMainScore,p.main_score_state.as_ref()),
        (Stage::CacheIndexerState,p.indexer_kv_state.as_ref()),(Stage::CacheIndexerScore,p.indexer_score_state.as_ref())] {
        if let Some(b)=b {out.push((stage,b,None));}
    }out
}
pub fn expected(cfg:&Deepseek4Config,capacity:usize,boundaries:&[(usize,bool)]) ->Result<Vec<ExpectedTensor>,String> {
    let plan=Deepseek4CachePlan::for_context(cfg,capacity).map_err(|e|e.to_string())?;
    let mut out=Vec::new();
    for &(position,restored) in boundaries {
        if ![64,128,144].contains(&position) || position>capacity || (restored && position!=64) {
            return Err("DS diagnostic boundary invalid".into());
        }
        for lane in 0..4 {for layer in &plan.layers {for (stage,p,rows) in roles(layer,position) {
            out.push(ExpectedTensor {at:at(lane,position,layer.layer_index,stage,restored),
                scalar:scalar(p.dtype)?,shape:valid_shape(p,rows)?});
        }}}
    }Ok(out)
}
fn actual(layer:&crate::inference::models::deepseek4::cache::LayerCache,stage:Stage)->Option<&MlxBuffer> {
    match stage {Stage::CacheWindow=>Some(&layer.window_kv),Stage::CacheCompressed=>layer.compressed_kv.as_ref(),
        Stage::CacheIndexer=>layer.indexer_kv.as_ref(),Stage::CacheMainKv=>layer.main_kv_state.as_ref(),
        Stage::CacheMainScore=>layer.main_score_state.as_ref(),Stage::CacheIndexerState=>layer.indexer_kv_state.as_ref(),
        Stage::CacheIndexerScore=>layer.indexer_score_state.as_ref(),_=>None}
}
pub fn after_checked_terminal(store:&mut Store,cfg:&Deepseek4Config,cache:&Deepseek4Cache,
    lane:usize,position:usize,restored:bool)->Result<(),String> {
    if lane>=4 || cache.position()!=position || cache.is_poisoned() || cache.graft_slots()!=0 {
        return Err("DS actual cache owner/cursor/poison/graft mismatch".into());
    }
    let plan=Deepseek4CachePlan::for_context(cfg,cache.capacity()).map_err(|e|e.to_string())?;
    if cache.layers().len()!=plan.layers.len() {return Err("DS actual layer count mismatch".into());}
    for (layer,p) in cache.layers().iter().zip(&plan.layers) {
        // Validate every optional role, including one with zero currently valid
        // append-only rows, before selecting the actual active extent.
        for (stage,descriptor) in [(Stage::CacheWindow,Some(&p.window_kv)),
            (Stage::CacheCompressed,p.compressed_kv.as_ref()),(Stage::CacheIndexer,p.indexer_kv.as_ref()),
            (Stage::CacheMainKv,p.main_kv_state.as_ref()),(Stage::CacheMainScore,p.main_score_state.as_ref()),
            (Stage::CacheIndexerState,p.indexer_kv_state.as_ref()),(Stage::CacheIndexerScore,p.indexer_score_state.as_ref())] {
            match (actual(layer,stage),descriptor) {
                (Some(b),Some(d)) if b.shape()==d.shape.as_slice() && b.dtype()==d.dtype=>{},
                (None,None)=>{},_=>return Err("DS actual cache role/layout differs from native plan".into())
            }
        }
        for (stage,descriptor,rows) in roles(p,position) {
            let b=actual(layer,stage).ok_or("DS active role missing")?;
            let shape=valid_shape(descriptor,rows)?;
            let elements=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).ok_or("DS active extent overflow")?;
            let view=b.slice_view(0,elements).with_shape(shape.clone()).map_err(|e|e.to_string())?;
            capture::after_checked_drain(store,at(lane,position,p.layer_index,stage,restored),&view,&shape,None,
                "actual DS verifier/head checked finish + published cursor; complete physical active cache extent")?;
        }
    }Ok(())
}
pub fn require_restore64_exact(tensors:&[Tensor])->Result<u64,String> {
    let before=tensors.iter().filter(|t|t.at.logical_step==64 && t.at.segment==0).collect::<Vec<_>>();
    let after=tensors.iter().filter(|t|t.at.logical_step==64 && t.at.segment==1).collect::<Vec<_>>();
    if before.is_empty() || before.len()!=after.len(){return Err("DS full exact64 paired extent missing".into());}
    let mut total=0u64;
    for source in before {
        let mut at=source.at.clone();at.segment=1;
        let matched=after.iter().filter(|t|t.at==at).collect::<Vec<_>>();
        if matched.len()!=1 {return Err("DS exact64 coordinate missing/ambiguous".into());}
        let target=matched[0];
        if source.scalar!=target.scalar || source.shape!=target.shape || source.logical_bytes==0
            || source.logical_bytes!=target.logical_bytes || source.sha256!=target.sha256 {
            return Err("DS restored active64 bytes differ".into());
        }
        total=total.checked_add(source.logical_bytes).ok_or("DS exact64 count overflow")?;
    }Ok(total)
}
#[cfg(test)]
#[path="endpoint_controls.rs"]
mod controls;
