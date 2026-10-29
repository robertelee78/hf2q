//! cfg(test) actual DeepSeek cache/compact-anchor buffer inventory. No GPU work.
use super::{Deepseek4Cache,Deepseek4CacheSnapshot};
use crate::inference::backing_observation::{observe,RoleSnapshot};
use mlx_native::MlxBuffer;
use anyhow::{ensure,Result};
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum Presence {Present,Absent,Unknown(&'static str)}
#[derive(Clone,Debug)]
pub struct FieldRole {pub field:String,pub presence:Presence,pub buffer:Option<RoleSnapshot>}
#[derive(Debug)]
pub struct DeepseekCacheRoles {
    pub owner:String,pub kind:&'static str,pub capacity:usize,pub position:usize,
    pub poisoned:Option<bool>,pub graft_slots:Option<usize>,
    pub compression_ratios:Vec<u32>,pub fields:Vec<FieldRole>,
    pub external_scopes:Vec<FieldRole>,
}
pub fn buffer_field(field:String,buffer:Option<&MlxBuffer>)->Result<FieldRole> {
    ensure!(field.len()<=256&&!field.is_empty(),"bounded DeepSeek owner/field name required");
    Ok(match buffer {Some(b)=>FieldRole{buffer:Some(observe(field.clone(),b)?),field,presence:Presence::Present},
        None=>FieldRole{field,presence:Presence::Absent,buffer:None}})
}
pub fn unknown_field(field:String,reason:&'static str)->FieldRole {FieldRole{field,presence:Presence::Unknown(reason),buffer:None}}
fn owner_check(owner:&str,layers:usize)->Result<()> {ensure!(!owner.is_empty()&&owner.len()<=128&&layers<=1024,"bounded DeepSeek owner/layer metadata required");Ok(())}
/// All actual allocated parent+view fields, including invisible append capacity.
/// No sum of logical views: window/compressed alias attention_kv's SAME root.
pub fn deepseek_cache_roles(owner:&str,cache:&Deepseek4Cache)->Result<DeepseekCacheRoles> {
    owner_check(owner,cache.layers.len())?;ensure!(cache.layers.len()==cache.plan.layers.len(),"DeepSeek plan/owner layer mismatch");
    let mut fields=Vec::new();
    for (index,layer) in cache.layers.iter().enumerate() {
        let prefix=format!("{owner}/layer/{index}");
        for (name,b) in [("attention_parent",Some(&layer.attention_kv)),("window_view",Some(&layer.window_kv)),
            ("compressed_view",layer.compressed_kv.as_ref()),("indexer_kv",layer.indexer_kv.as_ref()),
            ("main_kv_state",layer.main_kv_state.as_ref()),("main_score_state",layer.main_score_state.as_ref()),
            ("indexer_kv_state",layer.indexer_kv_state.as_ref()),("indexer_score_state",layer.indexer_score_state.as_ref())] {
            fields.push(buffer_field(format!("{prefix}/{name}"),b)?);
        }
    }
    Ok(DeepseekCacheRoles{owner:owner.into(),kind:"primary_allocated_cache_roots_and_alias_views",
        capacity:cache.plan.context_length,position:cache.next_position,poisoned:Some(cache.poisoned),graft_slots:Some(cache.graft_slots),
        compression_ratios:cache.plan.layers.iter().map(|l|l.compress_ratio).collect(),fields,
        external_scopes:external_scopes(owner)})
}
/// Compact anchor fields are independent copied parents; append-only global
/// compressed/indexer KV is not owned by this snapshot and is NOT recharged.
pub fn deepseek_anchor_roles(owner:&str,anchor:&Deepseek4CacheSnapshot)->Result<DeepseekCacheRoles> {
    owner_check(owner,anchor.layers.len())?;ensure!(anchor.layers.len()==anchor.plan.layers.len(),"DeepSeek anchor plan mismatch");
    let mut fields=Vec::new();
    for (index,layer) in anchor.layers.iter().enumerate() {
        let prefix=format!("{owner}/layer/{index}");
        for (name,b) in [("window_copy",Some(&layer.window_kv)),("main_kv_state_copy",layer.main_kv_state.as_ref()),
            ("main_score_state_copy",layer.main_score_state.as_ref()),("indexer_kv_state_copy",layer.indexer_kv_state.as_ref()),
            ("indexer_score_state_copy",layer.indexer_score_state.as_ref())] {fields.push(buffer_field(format!("{prefix}/{name}"),b)?);}
    }
    Ok(DeepseekCacheRoles{owner:owner.into(),kind:"compact_anchor_independent_copies_only",capacity:anchor.plan.context_length,
        position:anchor.next_position,poisoned:None,graft_slots:None,
        compression_ratios:anchor.plan.layers.iter().map(|l|l.compress_ratio).collect(),fields,
        external_scopes:external_scopes(owner)})
}
fn external_scopes(owner:&str)->Vec<FieldRole> {
    ["weights_cpu_staging","auxiliary_mtp_or_external_drafter","compressed_core_and_grouped_output_workspace",
     "ratio4_indexer_workspace","sparse_decode_gather_workspace","prefill_flash_workspaces","pool_and_pending_CB_owners"]
        .iter().map(|name|unknown_field(format!("{owner}/external/{name}"),"Not owned by this cache/snapshot type; requires actual separate owner join; zero credit forbidden")).collect()
}
/// Scope-complete only for PROVIDED owners; does not certify scheduler membership.
pub struct ProvidedDeepseekOwners<'a> {
    pub owner:&'a str,pub primary:&'a Deepseek4Cache,
    pub committed:Option<&'a Deepseek4CacheSnapshot>,pub pending:Option<&'a Deepseek4CacheSnapshot>,
    pub live_logits:Option<&'a MlxBuffer>,
}
#[derive(Debug)]pub struct DeepseekProvidedOwnerRoles {pub cache:DeepseekCacheRoles,pub anchor_presence:Vec<FieldRole>,pub anchors:Vec<DeepseekCacheRoles>,pub live_logits:FieldRole}
pub fn deepseek_provided_owner_roles(p:ProvidedDeepseekOwners<'_>)->Result<DeepseekProvidedOwnerRoles> {
    let cache=deepseek_cache_roles(p.owner,p.primary)?;let mut anchors=Vec::new();let mut anchor_presence=Vec::new();
    for (name,a) in [("committed_anchor",p.committed),("pending_anchor",p.pending)] {
        let field=format!("{}/{name}",p.owner);anchor_presence.push(FieldRole{field:field.clone(),presence:if a.is_some(){Presence::Present}else{Presence::Absent},buffer:None});
        if let Some(a)=a {anchors.push(deepseek_anchor_roles(&field,a)?);}
    }
    Ok(DeepseekProvidedOwnerRoles{cache,anchors,anchor_presence,live_logits:buffer_field(format!("{}/live_logits",p.owner),p.live_logits)?})
}
