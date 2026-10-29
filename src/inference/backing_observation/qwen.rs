use super::{observe,RoleSnapshot};
use crate::inference::models::qwen35::kv_cache::{HybridKvCache,FullAttnKvSlot};
use anyhow::{ensure,Result};
use mlx_native::MlxBuffer;
#[derive(Debug)]
pub struct QwenCacheRoles {
    pub coverage: &'static str,
    pub max_seq_len:u32,pub configured_slots:u32,
    pub full_attention_cursors:Vec<Vec<u32>>,pub mtp_cursors:Option<Vec<u32>>,
    /// Named physical roots are not necessarily CURRENT: parity is per slot.
    pub linear_ping_pong_parities:Vec<Vec<bool>>,
    pub roles:Vec<RoleSnapshot>,
}
fn push(out:&mut Vec<RoleSnapshot>,name:String,b:&MlxBuffer)->Result<()> {
    ensure!(out.len()<16384,"cache role snapshot bound exceeded");
    out.push(observe(name,b)?); Ok(())
}
fn full(out:&mut Vec<RoleSnapshot>,prefix:&str,s:&FullAttnKvSlot)->Result<()> {
    if let Some(b)=&s.k {push(out,format!("{prefix}/k_f32"),b)?;}
    if let Some(b)=&s.v {push(out,format!("{prefix}/v_f32"),b)?;}
    if let Some(t)=&s.tq {
        for (name,b) in [("k_packed",&t.k_packed),("k_norms",&t.k_norms),("v_packed",&t.v_packed),("v_norms",&t.v_norms)] {
            push(out,format!("{prefix}/{name}"),b)?;
        }
    }
    Ok(())
}
/// Actual entire allocated physical cache roots, not just cursor-active rows.
/// Rank indices explicitly match cache full_attn/linear_attn vectors, not model
/// layer indices. Both ping-pong and optional rollback roots remain charged.
/// Borrow while the owning worker is quiescent; does not synchronize GPU work.
/// Anchors/carry/scratch held OUTSIDE this cache are explicitly not covered.
pub fn qwen_cache_roles(cache:&HybridKvCache)->Result<QwenCacheRoles> {
    ensure!(cache.full_attn.len()<=1024 && cache.linear_attn.len()<=1024 && cache.n_seqs<=1024,
        "bounded cache role geometry required");
    let mut roles=Vec::new();
    for (rank,s) in cache.full_attn.iter().enumerate() { full(&mut roles,&format!("kv/full_rank/{rank}"),s)?; }
    if let Some(s)=&cache.mtp_slot {full(&mut roles,"kv/mtp",s)?;}
    for (rank,s) in cache.linear_attn.iter().enumerate() {
        ensure!(s.pp_flipped.len()<=1024,"bounded parity metadata required");
        for (name,b) in [("conv_physical0",&s.conv_state),("conv_physical1",&s.conv_state_scratch),
            ("recurrent_physical0",&s.recurrent),("recurrent_physical1",&s.recurrent_scratch)] {
            push(&mut roles,format!("recurrent/linear_rank/{rank}/{name}"),b)?;
        }
        if let Some(b)=&s.capture_states {push(&mut roles,format!("recurrent/linear_rank/{rank}/capture_states"),b)?;}
        if let Some(b)=&s.conv_capture_states {push(&mut roles,format!("recurrent/linear_rank/{rank}/conv_capture_states"),b)?;}
    }
    for s in &cache.full_attn {ensure!(s.current_len.len()<=1024,"bounded cursor metadata required");}
    if let Some(s)=&cache.mtp_slot {ensure!(s.current_len.len()<=1024,"bounded MTP cursor metadata required");}
    Ok(QwenCacheRoles{coverage:"qwen_allocated_cache_roots_only_not_anchors_carry_scratch",max_seq_len:cache.max_seq_len,configured_slots:cache.n_seqs,
        full_attention_cursors:cache.full_attn.iter().map(|s|s.current_len.clone()).collect(),
        mtp_cursors:cache.mtp_slot.as_ref().map(|s|s.current_len.clone()),
        linear_ping_pong_parities:cache.linear_attn.iter().map(|s|s.pp_flipped.clone()).collect(),roles})
}
