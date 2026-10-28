use super::*;
#[test]
fn source_plan_has_full_width_two_absolute_rows_and_no_last_row_alias() {
    let expected=expected();
    assert_eq!(expected.len(),258);assert_eq!(ARM_BYTES,5_283_840);
    let coordinates:BTreeSet<_>=expected.iter().map(|e|e.at.clone()).collect();
    assert_eq!(coordinates.len(),258);
    assert!(expected.iter().all(|e|e.shape==[1,5120] && e.scalar==Scalar::F32 &&
        e.at.agent=="agent-0" && [129,130].contains(&e.at.logical_step)));
    assert_eq!(plan::point(0,Stage::Input).unwrap(),0);
    assert_eq!(plan::point(63,Stage::Residual).unwrap(),127);
    assert_eq!(plan::point(63,Stage::LayerOutput).unwrap(),128);
    assert!(plan::point(64,Stage::LayerOutput).is_err());
    assert!(plan::point(1,Stage::Input).is_err());
    assert!(plan::point(0,Stage::Logits).is_err());
}
#[test]
fn source_axis_refuses_swapped_positions_and_wrong_current_input() {
    let wanted=[31,41];
    let positions=super::super::spec_decode::positions_for_range(129,4);
    assert_eq!(plan::select(true,&[31,41,0,100],&positions,wanted).unwrap(),Some(vec![129,130]));
    assert_eq!(plan::select(false,&[41],&[130;4],wanted).unwrap(),Some(vec![130]));
    assert!(plan::select(false,&[31],&[130;4],wanted).is_err());
    assert!(plan::select(true,&[41,31,0,100],&positions,wanted).is_err());
    let mut swapped=positions;swapped.swap(0,1);
    assert!(plan::select(true,&[31,41,0,100],&swapped,wanted).is_err());
    assert!(plan::select(true,&[31],&[129;4],wanted).is_err());
    assert_eq!(plan::select(false,&[7],&[128;4],wanted).unwrap(),None);
}
#[test]
#[ignore="192 requested GPU bytes; real ordered blit snapshot; root supervised fresh process"]
fn snapshot_blit_precedes_real_source_overwrite_without_extra_wait() {
    let _lock=crate::inference::hf2q_gpu_test_lock();
    let device=MlxDevice::new().expect("actual Metal required; no device-unavailable PASS");
    let mut source=device.alloc_buffer(64,DType::F32,vec![2,8]).unwrap();
    let snapshot=device.alloc_buffer(64,DType::F32,vec![2,8]).unwrap();
    let mut overwrite=device.alloc_buffer(64,DType::F32,vec![2,8]).unwrap();
    let values:Vec<f32>=(1..=16).map(|v|v as f32).collect();
    source.as_mut_slice::<f32>().unwrap().copy_from_slice(&values);
    overwrite.as_mut_slice::<f32>().unwrap().fill(0.0);
    let mut enc=device.command_encoder().unwrap();
    copy_rows(&mut enc,&source,&snapshot,2,8,0).unwrap();
    enc.blit_copy_bytes(&overwrite,0,&source,0,64).unwrap();
    enc.commit_and_wait_labeled("q1.target.snapshot.actual-copy-control").unwrap();
    assert_eq!(snapshot.as_logical_slice::<f32>().unwrap(),values.as_slice());
    assert!(source.as_logical_slice::<f32>().unwrap().iter().all(|&v|v.to_bits()==0));
    println!("Q1_TARGET_SNAPSHOT_COPY_COMPLETE actual_waits=1 requested_gpu_bytes=192 exact_values=16 source_overwritten=true");
}

#[test]
fn actual_point_guard_refuses_duplicate_missing_and_replaced_stage() {
    let mut seen=BTreeSet::new();
    plan::visit(&mut seen,0,Stage::Input).unwrap();
    assert!(plan::visit(&mut seen,0,Stage::Input).is_err());
    assert!(plan::complete(&seen).is_err());
    for layer in 0..64 {plan::visit(&mut seen,layer,Stage::Residual).unwrap();
        plan::visit(&mut seen,layer,Stage::LayerOutput).unwrap();}
    plan::complete(&seen).unwrap();
    seen.remove(&100);seen.insert(129);
    assert!(plan::complete(&seen).is_err());
}
