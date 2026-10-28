//! Actual scheduler/callback controls; no device or fabricated elapsed-time proof.
use super::*;
fn request() -> AdmitRequest {
    AdmitRequest {prompt_tokens:720,max_tokens:8192,kv_bytes_needed:0,prompt_kv_bytes:0,preferred_slot:None}
}
fn selected(s: &mut InflightBatchedScheduler) -> (SlotHandle,u32) {
    match s.step().unwrap() {
        SchedulerStep::Prefill {handle,n_tokens} => (handle,n_tokens),
        SchedulerStep::Mixed {prefill,n_prefill_tokens,..} => (prefill,n_prefill_tokens),
        other => panic!("expected pending prompt, got {other:?}"),
    }
}
#[test]
fn diagnostic128_four720_actual_callbacks_rotate_preserve_tail_and_generation() {
    let mut s=InflightBatchedScheduler::new(8,4);
    s.set_prefill_chunk_tokens(128);
    let handles:Vec<_>=(0..4).map(|_| s.admit(request()).unwrap().handle.unwrap()).collect();
    for round in 0..6 {
        for expected in &handles {
            let (handle,tokens)=selected(&mut s);
            assert_eq!(handle,*expected);assert_eq!(tokens,if round==5 {80} else {128});
            s.advance_after_prefill(handle,tokens);
            s.yield_prefill_turn(handle);
        }
    }
    match s.step().unwrap() {
        SchedulerStep::Decode {handles:active} => assert_eq!(active,handles),
        other=>panic!("all720 positions must finish exactly, got {other:?}"),
    }
    let retired=handles[0];s.release(retired);
    let fresh=s.admit(request()).unwrap().handle.unwrap();
    assert_eq!(fresh.slot_id,retired.slot_id);assert!(fresh.generation>retired.generation);
    s.advance_after_prefill(retired,720);s.yield_prefill_turn(retired);
    let (actual,tokens)=selected(&mut s);assert_eq!(actual,fresh);assert_eq!(tokens,128);
}
#[test]
fn diagnostic128_mixed_keeps_decode_handles_and_default2048_is_not_replaced() {
    let mut s=InflightBatchedScheduler::new(8,4);
    s.set_prefill_chunk_tokens(128);
    let decoded=s.admit(request()).unwrap().handle.unwrap();
    s.advance_after_prefill(decoded,720);
    let cold=s.admit(request()).unwrap().handle.unwrap();
    match s.step().unwrap() {
        SchedulerStep::Mixed {prefill,n_prefill_tokens,decode_handles}=>{
            assert_eq!(prefill,cold);assert_eq!(n_prefill_tokens,128);
            assert_eq!(decode_handles,vec![decoded]);
        }
        other=>panic!("mixed retains existing decode cohort, got {other:?}"),
    }
    let mut normal=InflightBatchedScheduler::new(8,4);
    normal.set_prefill_chunk_tokens(2048);
    normal.admit(request()).unwrap();
    assert_eq!(selected(&mut normal).1,720);
}
