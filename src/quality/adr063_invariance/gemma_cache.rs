//! Complete active selected Hybrid state, per slot/head (not padded capacity).
use super::{capture, receipt::*, store::Store};
use crate::{inference::models::gemma4::kv_cache::MultiSeqHybridKvBuffers,
    serve::config::Gemma4Config};
use mlx_native::DType;
pub fn at(lane:usize,position:usize,layer:usize,stage:Stage,head:usize,restored:bool)->Coordinate {
    Coordinate {logical_step:position as u64,layer:layer as u32,stage,
        agent:format!("agent-{lane}"),logical_row_start:0,
        segment:head as u32+if restored {1000}else{0}}
}
pub fn expected(cfg:&Gemma4Config,lanes:&[usize],positions:&[(usize,bool)]) -> Result<Vec<ExpectedTensor>,String> {
    if cfg.sliding_window<144 || cfg.num_hidden_layers==0 {return Err("selected no-wrap144 Gemma geometry required".into());}
    let mut out=Vec::new();
    for &lane in lanes {for &(position,restored) in positions {
        if lane>=4 || ![64,128,144].contains(&position) || (restored && position!=64) {
            return Err("Gemma cache boundary scope".into());
        }
        for layer in 0..cfg.num_hidden_layers {
            let hd=cfg.head_dim_for_layer(layer);let nkv=cfg.num_kv_heads_for_layer(layer);
            if ![256,512].contains(&hd) || nkv>=1000 {return Err("selected Gemma head geometry unsupported".into());}
            for head in 0..nkv {for (stage,scalar,width) in [
                (Stage::CacheKey,Scalar::F16,hd),(Stage::CacheValuePacked,Scalar::U8,hd),
                (Stage::CacheValueNorm,Scalar::F32,hd/256)] {
                out.push(ExpectedTensor {at:at(lane,position,layer,stage,head,restored),scalar,
                    shape:vec![1,position,width]});
            }}
        }
    }}
    Ok(out)
}
pub fn after_checked_return(store:&mut Store,cfg:&Gemma4Config,
    cache:&[MultiSeqHybridKvBuffers],lane:usize,position:usize,restored:bool)->Result<(),String> {
    if cache.len()!=cfg.num_hidden_layers || cfg.sliding_window<144 {return Err("Gemma layer/no-wrap mismatch".into());}
    for (layer,c) in cache.iter().enumerate() {
        let hd=cfg.head_dim_for_layer(layer);let nkv=cfg.num_kv_heads_for_layer(layer);
        if lane>=c.n_seqs as usize || c.n_seqs!=4 || c.capacity<position
            || c.seq_lens.get(lane).copied()!=Some(position as u32)
            || c.is_sliding==cfg.is_full_attention(layer)
            || c.k.dtype()!=DType::F16 || c.v_packed.dtype()!=DType::U8
            || c.v_norms.dtype()!=DType::F32 || c.norms_per_pos!=hd/256
            || c.bf16_xlen_k.is_some() || c.bf16_xlen_v.is_some() {
            return Err("actual Gemma Hybrid scaffold/profile/cursor mismatch".into());
        }
        for (buffer,stage,width) in [(&c.k,Stage::CacheKey,hd),
            (&c.v_packed,Stage::CacheValuePacked,hd),(&c.v_norms,Stage::CacheValueNorm,hd/256)] {
            if buffer.shape()!=[4,nkv,c.capacity,width] {return Err("Gemma actual slot/head/cache axes mismatch".into());}
            for head in 0..nkv {
                let elements=position.checked_mul(width).ok_or("Gemma active cache overflow")?;
                let offset=(lane*nkv+head).checked_mul(c.capacity).and_then(|n|n.checked_mul(width))
                    .and_then(|n|n.checked_mul(buffer.dtype().size_of())).ok_or("Gemma slot offset overflow")?;
                let view=buffer.slice_view(offset as u64,elements).with_shape(vec![1,position,width])
                    .map_err(|e|e.to_string())?;
                capture::after_checked_drain(store,at(lane,position,layer,stage,head,restored),
                    &view,&[1,position,width],None,"actual Gemma slot-aware checked return; cursor==boundary")?;
            }
        }
    }
    Ok(())
}
