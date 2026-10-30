//! Actual tiny typed-cache constructors, NO model or forward. Root executes only.
use super::*;
use crate::inference::models::deepseek4::{Deepseek4Config,cache::{Deepseek4Cache,Deepseek4CachePlan}};
use std::collections::{BTreeMap,BTreeSet};
pub(crate) fn tiny_config()->Deepseek4Config {
    // Same head/indexer/compressor geometry; only3layers and C256, ratios0/4/128.
    Deepseek4Config{num_hidden_layers:3,hidden_size:4096,hidden_size_out:16384,
        max_position_embeddings:256,vocab_size:129280,num_attention_heads:64,num_key_value_heads:1,
        head_dim:512,rope_head_dim:64,rope_theta:10000.,rope_factor:16.,original_context_length:65536,
        yarn_beta_fast:32.,yarn_beta_slow:1.,q_lora_rank:1024,o_lora_rank:1024,output_groups:8,
        sliding_window:128,compress_ratios:vec![0,4,128],compress_rope_theta:160000.,
        index_num_heads:64,index_head_dim:128,index_top_k:512,rms_norm_eps:1e-6,
        num_experts:256,num_experts_per_tok:6,num_shared_experts:1,expert_intermediate_size:2048,
        route_scale:1.5,normalize_topk:true,swiglu_clamp_experts:vec![10.;3],swiglu_clamp_shared:vec![10.;3],
        hyper_connection_count:4,hyper_connection_sinkhorn_iterations:20,hyper_connection_epsilon:1e-6,hash_layer_count:3}
}
fn roots(fields:&[FieldRole])->BTreeMap<usize,usize> {
    let mut roots=BTreeMap::new();
    for f in fields {match (&f.presence,&f.buffer){
        (Presence::Present,Some(r))=>{assert!(r.native_origin_known);assert_eq!(r.backing.provenance,mlx_native::BufferProvenance::Owned);
            match r.backing.origin {mlx_native::RootOriginSnapshot::KnownNative{id,requested_bytes}=>{
                assert_eq!(requested_bytes,r.backing.metal_root_bytes);
                if let Some(old)=roots.insert(id,requested_bytes){assert_eq!(old,requested_bytes);}
            },_=>panic!("unknown actual native root")}}
        (Presence::Absent,None)=>{},_=>panic!("unexpected cache buffer presence")}}
    roots
}
fn role<'a>(r:&'a DeepseekCacheRoles,suffix:&str)->&'a RoleSnapshot {
    r.fields.iter().find(|f|f.field.ends_with(suffix)).unwrap().buffer.as_ref().unwrap()
}
#[test]
fn actual_ds_plan_matches_finite_three_ratio_payload_ledger() {
    let plan=Deepseek4CachePlan::for_context(&tiny_config(),256).unwrap();
    assert_eq!(plan.layers.iter().map(|l|l.resident_bytes).collect::<Vec<_>>(),vec![131072,294912,657408]);
    assert_eq!(plan.resident_bytes,1_083_392);
    // Snapshot full128-row window per layer +all existing recurrent states.
    let anchor=3*128*512*2 +2*(1*8*1024*4)+2*(1*8*256*4)+2*(1*128*512*4);
    assert_eq!(anchor,999_424);
    assert_eq!(2*plan.resident_bytes+4*anchor,6_164_480);
    assert!(6_164_480+56*65536+1024*1024<16*1024*1024);
}
#[test]
#[ignore="fresh-process <=16MiB DeepSeek actual cache+anchor roles; root supervised only"]
fn actual_deepseek_two_cache_owners_all_ratios_aliases_and_anchors() {
    use mlx_native::{MlxDevice,NativeRootAuthority,RootCreationLimits};
    const REQUESTED:usize=6_164_480;const ROOTS:usize=56;
    assert!(REQUESTED+ROOTS*65536+1024*1024<16*1024*1024); // PRE-DEVICE named-user bound, not physical.
    let cfg=tiny_config();let plan=Deepseek4CachePlan::for_context(&cfg,256).unwrap();assert_eq!(plan.resident_bytes,1_083_392);
    let authority=NativeRootAuthority::install_early(RootCreationLimits{max_attempted_requested_bytes:REQUESTED,max_root_attempts:ROOTS,
        max_raw_constructor_attempts:ROOTS,max_registered_devices:1,max_rope_cache_entries:1}).unwrap();
    let counters=(mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count());
    let device=MlxDevice::new().expect("actual Metal required; no skip");authority.register_device(&device).unwrap();
    let mut caches=Vec::new();let mut committed=Vec::new();let mut pending=Vec::new();
    for owner in 0..2 {
        // Initialized allocate, not lazy/unwritten storage; CPU snapshots read no unwritten tail.
        let mut cache=Deepseek4Cache::allocate(&plan,device.clone()).unwrap();
        for pos in 0..=owner {cache.commit_step(pos).unwrap();} // Metadata-only fixture, not executed token proof.
        committed.push(cache.snapshot().unwrap());pending.push(cache.snapshot().unwrap());caches.push(cache);
    }
    let mut all=BTreeMap::new();
    for owner in 0..2 {
        let label=format!("tiny/owner{owner}");
        let report=deepseek_provided_owner_roles(ProvidedDeepseekOwners{owner:&label,primary:&caches[owner],committed:Some(&committed[owner]),pending:Some(&pending[owner]),live_logits:None}).unwrap();
        let cache=&report.cache;assert_eq!((cache.capacity,cache.position),(256,owner+1));assert_eq!(cache.poisoned,Some(false));assert_eq!(cache.graft_slots,Some(0));
        assert_eq!(cache.compression_ratios,vec![0,4,128]);assert_eq!(cache.fields.len(),24);
        let expected=(0..3).flat_map(|layer|["attention_parent","window_view","compressed_view","indexer_kv","main_kv_state","main_score_state","indexer_kv_state","indexer_score_state"].into_iter().map(move |name|format!("tiny/owner{owner}/layer/{layer}/{name}"))).collect::<BTreeSet<_>>();
        assert_eq!(cache.fields.iter().map(|f|f.field.clone()).collect::<BTreeSet<_>>(),expected);
        assert_eq!(cache.fields.iter().filter(|f|f.presence==Presence::Present).count(),15);
        assert_eq!(cache.fields.iter().filter(|f|f.presence==Presence::Absent).count(),9);
        assert!(cache.external_scopes.iter().all(|f|matches!(f.presence,Presence::Unknown(_))&&f.buffer.is_none()));
        let ids=roots(&cache.fields);assert_eq!(ids.len(),10);assert_eq!(ids.values().sum::<usize>(),1_083_392);
        assert!(ids.keys().all(|id|!all.contains_key(id)));all.extend(ids);
        for layer in 0..3 {
            let parent=role(cache,&format!("layer/{layer}/attention_parent"));let window=role(cache,&format!("layer/{layer}/window_view"));
            assert_eq!(parent.backing.origin,window.backing.origin);assert_eq!(window.backing.view_byte_offset,0);assert_eq!(window.backing.logical_bytes,131072);
            if layer!=0 {let compressed=role(cache,&format!("layer/{layer}/compressed_view"));assert_eq!(compressed.backing.origin,parent.backing.origin);
                assert_eq!(compressed.backing.view_byte_offset,131072);assert_eq!(compressed.backing.logical_bytes,if layer==1{65536}else{2048});}
        }
        assert_eq!(report.anchors.len(),2);assert!(report.anchor_presence.iter().all(|f|f.presence==Presence::Present));
        assert_eq!(report.live_logits.presence,Presence::Absent);
        for anchor in &report.anchors {
            assert_eq!(anchor.fields.len(),15);
            let expected=(0..3).flat_map(|layer|["window_copy","main_kv_state_copy","main_score_state_copy","indexer_kv_state_copy","indexer_score_state_copy"].into_iter().map(move |name|format!("{}/layer/{layer}/{name}",anchor.owner))).collect::<BTreeSet<_>>();
            assert_eq!(anchor.fields.iter().map(|f|f.field.clone()).collect::<BTreeSet<_>>(),expected);
            let ids=roots(&anchor.fields);assert_eq!(ids.len(),9);assert_eq!(ids.values().sum::<usize>(),999424);
            assert_eq!(anchor.position,owner+1);assert!(ids.keys().all(|id|!all.contains_key(id)));all.extend(ids);
        }
        for layer in caches[owner].layers() {
            for score in [layer.main_score_state.as_ref(),layer.indexer_score_state.as_ref()].into_iter().flatten(){
                assert!(score.as_slice::<f32>().unwrap().iter().all(|x|x.to_bits()==f32::NEG_INFINITY.to_bits()));
            }
        }
        let clone=caches[owner].layers()[1].compressed_kv.as_ref().unwrap().clone();
        assert_eq!(clone.backing_snapshot().unwrap().origin,role(cache,"layer/1/attention_parent").backing.origin);
        // Retained metadata changes do not turn capacity into active-only accounting.
        assert_eq!(roots(&deepseek_cache_roles(&label,&caches[owner]).unwrap().fields).len(),10);
        println!("DS_BACKING_ROLES owner={owner} cache_fields=24 present=15 absent=9 roots=10 cache_requested=1083392 anchors=2 anchor_roots_each=9 anchor_requested_each=999424 external_unknown=true");
    }
    assert_eq!(all.len(),56);assert_eq!(all.values().sum::<usize>(),REQUESTED);
    let history=authority.snapshot().unwrap();assert_eq!((history.successful_roots,history.attempted_requested_bytes),(ROOTS,REQUESTED));
    // Actual anchor ownership remains after primary caches drop. No release credit.
    drop(caches);assert_eq!(deepseek_anchor_roles("retained/anchor",&committed[0]).unwrap().fields.len(),15);
    assert_eq!((mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count()),counters);
    println!("DS_BACKING_ROLES_COMPLETE owners=2 cache_roots=20 anchor_roots=36 requested_bytes=6164480 no_gpu_commands=true physical_credit_bytes=0 q5_qualified=false");
}
