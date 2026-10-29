//! Fresh-process actual allocated-cache traversal, not model/forward correctness.
use super::*;
use crate::inference::models::qwen35::{default_layer_types,Qwen35Config,Qwen35Variant};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use std::collections::BTreeSet;
fn tiny_config()->Qwen35Config {
    Qwen35Config{variant:Qwen35Variant::Dense,hidden_size:512,num_hidden_layers:2,
        num_attention_heads:2,num_key_value_heads:1,head_dim:256,
        linear_num_key_heads:1,linear_num_value_heads:1,
        linear_key_head_dim:128,linear_value_head_dim:128,linear_conv_kernel_dim:4,
        full_attention_interval:2,layer_types:default_layer_types(2,2),
        partial_rotary_factor:0.25,rope_theta:1e7,rotary_dim:64,
        mrope_section:[11,11,10,0],mrope_interleaved:true,rms_norm_eps:1e-6,
        max_position_embeddings:8,vocab_size:256,attn_output_gate:true,
        mtp_num_hidden_layers:1,mtp_use_dedicated_embeddings:false,
        intermediate_size:Some(1024),moe:None}
}
fn expected(tq:bool)->BTreeSet<String> {
    let mut names=BTreeSet::new();
    for prefix in ["kv/full_rank/0","kv/mtp"] {
        let suffixes:&[&str]=if tq {&["k_packed","k_norms","v_packed","v_norms"]} else {&["k_f32","v_f32"]};
        for suffix in suffixes {names.insert(format!("{prefix}/{suffix}"));}
    }
    for suffix in ["conv_physical0","conv_physical1","recurrent_physical0","recurrent_physical1","capture_states","conv_capture_states"] {
        names.insert(format!("recurrent/linear_rank/0/{suffix}"));
    }
    names
}
fn known_ids(snapshot:&QwenCacheRoles)->BTreeSet<usize> {
    snapshot.roles.iter().map(|r| {
        assert!(r.native_origin_known);
        assert_eq!(r.backing.provenance,BufferProvenance::Owned);
        assert!(r.backing.no_copy.is_none());
        assert_eq!(r.backing.view_byte_offset,0);
        assert_eq!(r.backing.logical_bytes,r.backing.metal_root_bytes);
        match r.backing.origin {mlx_native::RootOriginSnapshot::KnownNative{id,requested_bytes}=>{
            assert_eq!(requested_bytes,r.backing.metal_root_bytes);id},_=>panic!("missing actual authority origin")}
    }).collect()
}
#[test]
#[ignore = "fresh-process <=8MiB actual Qwen F32/TQ/MTP/recurrent traversal; root supervised only"]
fn actual_tiny_qwen_cache_allocation_roles_cursors_and_aliases() {
    use mlx_native::{NativeRootAuthority,RootCreationLimits};
    // PRE-DEVICE source ledger, two caches live together. C8,N2,oneKVhead,D256;
    // one base full-attention+one MTP slot; one LA rank with Dk=Dv128,Hv1.
    const F32_KV:usize=2*2*(2*1*8*256*4); // 65,536
    const TQ_KV:usize=2*(2*(2*1*8*256)+2*(2*1*8*4)); // 16,640
    const LA:usize=2*(128*128*1*2*4)+2*(384*3*2*4); // 280,576
    const CAPTURE:usize=(128*128*1*2*2*4)+(2*2*3*384*4); // 280,576
    const REQUESTED:usize=F32_KV+TQ_KV+2*LA+2*CAPTURE;
    assert_eq!(REQUESTED,1_204_480);
    //24 parents, pessimistic <=64KiB rounding allowance each +256KiB metadata.
    // This is a finite test-userdata upper, NOT process/physical admission.
    assert!(REQUESTED+24*65536+256*1024<8*1024*1024);
    let counters_before=(mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count());
    let cfg=tiny_config();
    let authority=NativeRootAuthority::install_early(RootCreationLimits{
        max_attempted_requested_bytes:REQUESTED,max_root_attempts:24,max_raw_constructor_attempts:24,
        max_registered_devices:1,max_rope_cache_entries:1}).unwrap();
    let device=MlxDevice::new().expect("actual Metal required; no skip");
    authority.register_device(&device).unwrap();
    let mut f32_cache=HybridKvCache::new_with_options(&cfg,&device,8,2,false).unwrap();
    f32_cache.ensure_la_capture(&cfg,&device,2).unwrap();
    let mut tq_cache=HybridKvCache::new_with_options(&cfg,&device,8,2,true).unwrap();
    tq_cache.ensure_la_capture(&cfg,&device,2).unwrap();
    let mut all_ids=BTreeSet::new();
    for (tq,cache,count,bytes) in [(false,&mut f32_cache,10,F32_KV+LA+CAPTURE),(true,&mut tq_cache,14,TQ_KV+LA+CAPTURE)] {
        assert_eq!(cache.full_attn.len(),1);assert_eq!(cache.linear_attn.len(),1);assert!(cache.mtp_slot.is_some());
        let before=qwen_cache_roles(cache).unwrap(); // ACTUAL function under test.
        assert_eq!(before.coverage,"qwen_allocated_cache_roots_only_not_anchors_carry_scratch");
        assert_eq!((before.max_seq_len,before.configured_slots),(8,2));
        assert_eq!(before.roles.len(),count);
        assert_eq!(before.full_attention_cursors,vec![vec![0,0]]);
        assert_eq!(before.mtp_cursors,Some(vec![0,0]));
        assert_eq!(before.linear_ping_pong_parities,vec![vec![false,false]]);
        assert_eq!(before.roles.iter().map(|r|r.role.clone()).collect::<BTreeSet<_>>(),expected(tq));
        assert_eq!(before.roles.iter().map(|r|r.backing.logical_bytes).sum::<usize>(),bytes);
        let ids=known_ids(&before);assert_eq!(ids.len(),count);
        assert!(all_ids.is_disjoint(&ids));all_ids.extend(ids.iter().copied());
        // Distinct allocation roles cannot accidentally collapse ping-pong roots.
        for r in &before.roles {
            if r.role.ends_with("packed") {assert_eq!(r.dtype,"U8");}
            else {assert_eq!(r.dtype,"F32");}
        }
        // Metadata-only fixture setup, not a claim that tokens were executed.
        cache.full_attn[0].current_len=vec![1,2];
        cache.mtp_slot.as_mut().unwrap().current_len=vec![3,4];
        cache.linear_attn[0].pp_flipped=vec![false,true];
        let after=qwen_cache_roles(cache).unwrap();
        assert_eq!(after.full_attention_cursors,vec![vec![1,2]]);
        assert_eq!(after.mtp_cursors,Some(vec![3,4]));
        assert_eq!(after.linear_ping_pong_parities,vec![vec![false,true]]);
        assert_eq!(known_ids(&after),ids);
        for (a,b) in before.roles.iter().zip(&after.roles) {
            assert_eq!(a.role,b.role);assert_eq!(a.shape,b.shape);assert_eq!(a.backing,b.backing);
        }
        // Known aliases preserve actual origin/parent without creating a root.
        let physical=&cache.linear_attn[0].recurrent;
        let alias=physical.clone();let a=observe("alias/recurrent".into(),&alias).unwrap();
        assert_eq!(a.backing,physical.backing_snapshot().unwrap());
        let id=match a.backing.origin {mlx_native::RootOriginSnapshot::KnownNative{id,..}=>id,_=>panic!("unknown clone")};
        assert!(ids.contains(&id));assert_eq!(authority.snapshot().unwrap().successful_roots,24);
        println!("HF_QWEN_CACHE_ROLES tq={tq} roles={count} distinct_roots={count} requested_bytes={bytes} aliases_create_roots=false cursor_parity_separate=true");
    }
    assert_eq!(all_ids.len(),24);
    let history=authority.snapshot().unwrap();assert_eq!(history.successful_roots,24);
    assert_eq!(history.attempted_requested_bytes,REQUESTED);
    assert_eq!((mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count()),counters_before);
    println!("HF_QWEN_CACHE_ROLES_COMPLETE roles=24 roots=24 requested_bytes=1204480 f32_tq_mtp_pingpong_capture=true gpu_dispatches=0 physical_credit_bytes=0 q5_qualified=false");
}
