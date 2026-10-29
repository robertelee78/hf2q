//! Actual owning adapter. Metadata only; call on quiescent owning worker.
//! Session-vector indices are NOT invented scheduler handles/generations.
use super::Deepseek4LoadedModel;
use crate::inference::backing_observation::{deepseek_provided_owner_roles,ProvidedDeepseekOwners,DeepseekProvidedOwnerRoles,Presence,FieldRole};
use crate::inference::models::deepseek4::cache::backing_roles::unknown_field;
use anyhow::{ensure,Result};
#[derive(Debug)]pub struct LoadedDeepseekRoles {
    pub current_loaded:DeepseekProvidedOwnerRoles,pub session_presence:Presence,
    pub sessions:Vec<DeepseekProvidedOwnerRoles>,pub unresolved:Vec<FieldRole>,
}
pub fn actual_loaded_deepseek_roles(loaded:&Deepseek4LoadedModel)->Result<LoadedDeepseekRoles> {
    let current_loaded=deepseek_provided_owner_roles(ProvidedDeepseekOwners{owner:"deepseek/loaded/current",
        primary:&loaded.cache,committed:loaded.turn_anchor.as_ref(),pending:loaded.pending_turn_anchor.as_ref(),live_logits:loaded.live_logits.as_ref()})?;
    let mut sessions=Vec::new();let session_presence=if let Some(owners)=loaded.slot_sessions.as_ref(){
        ensure!(owners.len()<=4,"selected DeepSeek <=4 session ownership bound");
        for (index,s) in owners.iter().enumerate(){let owner=format!("deepseek/retained_session_index/{index}");
            sessions.push(deepseek_provided_owner_roles(ProvidedDeepseekOwners{owner:&owner,primary:&s.cache,
                committed:s.turn_anchor.as_ref(),pending:s.pending_turn_anchor.as_ref(),live_logits:s.live_logits.as_ref()})?);
        }Presence::Present
    }else{Presence::Absent};
    // A swapped-out/bootstrap cache is included if it is ACTUALLY held above;
    // no assumption that current loaded cache equals/aliases one retained slot.
    let unresolved=["model_weight_and_CPU_owners","transient_pool_and_pending_commands","all_forward_local_workspaces",
        "auxiliary_model_or_MTP_external_owner","CPU_token_vectors_and_disk_persistor","scheduler_slot_generation_authority","OS_partition"]
        .iter().map(|name|unknown_field(format!("deepseek/external/{name}"),"Separate actual owner/receipt required; this adapter gives no zero or physical credit")).collect();
    Ok(LoadedDeepseekRoles{current_loaded,session_presence,sessions,unresolved})
}
