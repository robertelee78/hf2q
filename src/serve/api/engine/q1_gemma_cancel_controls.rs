//! Actual one-shot producer guard controls; no synthetic model success.
use super::*;
#[test]
fn gemma_cancel_hook_requires_checked_success_exact_owner_and_cursor() {
    let handle=SlotHandle {slot_id:SlotId(2),generation:7};
    let called=Rc::new(Cell::new(0));let counter=called.clone();
    let armed=arm(handle,128,move||counter.set(counter.get()+1)).unwrap();
    after_actual_advance(false,handle,Some(128)).unwrap();
    assert!(!armed.fired.get());assert_eq!(called.get(),0);
    assert!(after_actual_advance(true,SlotHandle {generation:8,..handle},Some(128)).is_err());
    assert!(after_actual_advance(true,SlotHandle {slot_id:SlotId(1),..handle},Some(128)).is_err());
    assert!(after_actual_advance(true,handle,None).is_err());
    assert!(after_actual_advance(true,handle,Some(64)).is_err());
    assert_eq!(called.get(),0);
    after_actual_advance(true,handle,Some(128)).unwrap();
    assert!(armed.fired.get());assert_eq!(called.get(),1);
    after_actual_advance(true,handle,Some(128)).unwrap();
    assert_eq!(called.get(),1);drop(armed);
}
#[test]
fn gemma_cancel_hook_drop_disarms_and_rejects_nested_owner() {
    let h=SlotHandle {slot_id:SlotId(0),generation:1};
    let called=Rc::new(Cell::new(false));let value=called.clone();
    let armed=arm(h,128,move||value.set(true)).unwrap();
    assert!(arm(h,128,||{}).is_err());drop(armed);
    after_actual_advance(true,h,Some(128)).unwrap();
    assert!(!called.get());
}
#[test]
fn gemma_cancel_admits_every_lane_on_its_own_slot_with_one_in_flight() {
    for lane in 0..4u32 {
        let mut scheduler=InflightBatchedScheduler::new(4,4);
        let handle=admit_lane(&mut scheduler,SlotId(lane)).unwrap();
        assert_eq!(handle.slot_id,SlotId(lane));
        assert_eq!(scheduler.stats().in_flight_slots,1);
    }
}
