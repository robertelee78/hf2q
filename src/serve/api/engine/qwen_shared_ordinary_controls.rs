use super::*;
fn rows()->Vec<graph::Row> {(0..3).map(|i|graph::Row {request:i as u64,
    handle:SlotHandle {slot_id:crate::serve::multi_seq_kv::SlotId(i),generation:7},position:128+i,token:100+i}).collect()}
fn maps(rows:&[graph::Row])->Vec<graph::OperatorMap> {
    let p=|layer,name,rs:Vec<graph::Row>|graph::OperatorMap {layer,operator:name,physical_rows:rs.len(),rows:rs};
    let mut out=vec![p(None,"packed_embedding_gather_not_weight_matmul",rows.to_vec())];
    for l in 0..64 {
        for r in rows {out.push(p(Some(l),"singleton_entire_attention_including_QKV_output",vec![r.clone()]));}
        out.push(p(Some(l),"packed_postattn_RMS_dense_FFN_residual",rows.to_vec()));
    }
    out.push(p(None,"packed_output_RMS_full_vocabulary_head",rows.to_vec()));out
}
#[test]fn effective_actual_dispatch_bits_are_eight_not_just_tq_active() {
    assert!(effective_tq8(true,8));assert!(effective_tq8(true,0));
    for b in [5,6] {assert!(!effective_tq8(true,b));}
    assert!(!effective_tq8(false,8));
}
#[test]fn actual_operator_map_validation_retains_singletons_and_real_request_zero() {
    let r=rows();let good=maps(&r);validate_maps(&good,&r).unwrap();assert_eq!(r[0].request,0);
    let mut bad=maps(&r);bad[1].physical_rows=3;assert!(validate_maps(&bad,&r).is_err());
    let mut bad=maps(&r);bad[1].rows[0].handle.generation=8;assert!(validate_maps(&bad,&r).is_err());
    let mut bad=maps(&r);bad.pop();assert!(validate_maps(&bad,&r).is_err());
    let mut bad=maps(&r);bad[4].operator="whole_body_b4";assert!(validate_maps(&bad,&r).is_err());
}
#[test]fn actual_existing_fatal_fanout_owns_all_four_waiters_without_origin_double_release() {
    let mut scheduler=InflightBatchedScheduler::new(4,4);let mut slots=Vec::new();let mut receivers=Vec::new();
    for _ in 0..4 {
        let admitted=scheduler.admit(AdmitRequest {prompt_tokens:1,max_tokens:2,..AdmitRequest::default()}).unwrap();
        let h=admitted.handle.unwrap();let (tx,rx)=oneshot::channel();receivers.push(rx);
        let state=crate::serve::api::engine_qwen35::ordinary_prepared::controls::host_state(h.slot_id);
        slots.push(Some((Qwen35SlotWork::Decode(state),SlotReply::Unary(tx),h)));
    }
    let supervisor=EngineSupervisor::new();let (tx,mut rx)=mpsc::channel(1);drop(tx);
    let stats=Arc::new(Mutex::new(scheduler.stats()));let mut pending=VecDeque::new();
    fail_qwen35_worker_after_gpu_fatal(&supervisor,&mut rx,&mut pending,&mut slots,&mut scheduler,&stats,
        Qwen35FatalFailure {handle:None,reply:Qwen35FatalReply::None,
            error:anyhow::anyhow!("synthetic shared failure; no GPU"),kind:Qwen35WorkerFatalKind::Gpu});
    for rx in receivers {assert!(rx.blocking_recv().unwrap().is_err());}
    assert!(slots.iter().all(Option::is_none));assert_eq!(scheduler.stats().completed_total,4);
    assert!(!supervisor.is_healthy());
}
