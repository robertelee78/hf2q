use super::*;
fn row()->Vec<f32>{let mut v=vec![-2.;VOCAB];v[4861]=3.;v}
#[test]
fn actual_selection_checker_never_substitutes_a_cpu_token(){
    let v=row();assert_eq!(inspect(&v,4861).unwrap(),(true,false));
    assert!(inspect(&v,496).is_err());assert!(inspect(&v,VOCAB as u32).is_err());
    assert!(inspect(&v[..VOCAB-1],4861).is_err());
}
#[test]
fn equal_max_and_eos_are_ineligible_before_becoming_canonical_inputs(){
    let mut v=row();v[5000]=3.;let d=decision(0,127,4861,&v).unwrap();
    assert!(!d.unique_max);assert!(require_canonical(&d).is_err());
    v[4861]=-2.;v[5000]=-2.;v[1]=4.;
    assert!(require_canonical(&decision(0,127,1,&v).unwrap()).is_err());
    v[1]=-2.;v[106]=4.;assert!(require_canonical(&decision(0,142,106,&v).unwrap()).is_err());
}
#[test]
fn full_vocabulary_nonfinite_is_not_hidden_by_selected_maximum(){
    let mut v=row();v[VOCAB-1]=f32::NAN;assert!(inspect(&v,4861).is_err());
    v[VOCAB-1]=f32::INFINITY;assert!(inspect(&v,4861).is_err());
    v[VOCAB-1]=f32::NEG_INFINITY;assert!(inspect(&v,4861).is_err());
}
#[test]
fn preparation_prefix_preserves_first128_and_has_distinct_digest(){
    let w=Workload {family:Family::Gemma4,vocab_size:VOCAB as u32,
        lanes:(0..4).map(|lane|(0..144).map(|i|1000+lane*144+i).collect()).collect()};
    let p=prefixes(&w).unwrap();for lane in 0..4{assert_eq!(p[lane],w.lanes[lane][..128]);}
    let a=acquisition::preparation_prefix_digest(Family::Gemma4,&p,VOCAB as u32).unwrap();
    assert_ne!(a,acquisition::tokens_digest(&w).unwrap());
    let mut wrong=p.clone();wrong[0][127]+=1;
    assert_ne!(a,acquisition::preparation_prefix_digest(Family::Gemma4,&wrong,VOCAB as u32).unwrap());
    assert!(acquisition::preparation_prefix_digest(Family::Deepseek4,&p,VOCAB as u32).is_err());
    wrong[1]=wrong[0].clone();assert!(acquisition::preparation_prefix_digest(Family::Gemma4,&wrong,VOCAB as u32).is_err());
}
#[test]
fn preparation_quota_coordinates_and_carry_are_explicit(){
    assert_eq!(PAYLOAD,71303168);assert_eq!(4*17,68);assert_eq!(128+16,144);
    let d=decision(3,143,4861,&row()).unwrap();assert_eq!(d.input_position,143);
    assert_eq!(d.row_sha256,hash(&row().iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()));
}
#[test]
fn output_preflight_refuses_existing_or_repository_paths(){
    assert!(output_preflight(Path::new("relative")).is_err());
    assert!(output_preflight(&std::env::current_dir().unwrap()).is_err());
}
#[test]
fn actual_chain_gate_rejects_old_forced_input_missing_rows_carry_and_prefix_changes(){
    let prompts=(0..4).map(|n|vec![1000+n;128]).collect::<Vec<_>>();
    let mut lanes=prompts.clone();for v in &mut lanes{v.extend(std::iter::repeat_n(4861,16));}
    let mut ds=(0..4).flat_map(|lane|(127..144).map(move|pos|Decision {lane,input_position:pos,
        gpu_selected:4861,unique_max:true,eos:false,row_sha256:"a".repeat(64)})).collect::<Vec<_>>();
    require_chain(&prompts,&lanes,&ds,&[4861;4]).unwrap();
    lanes[0][128]=496;assert!(require_chain(&prompts,&lanes,&ds,&[4861;4]).is_err());lanes[0][128]=4861;
    assert!(require_chain(&prompts,&lanes,&ds,&[0;4]).is_err());
    ds[1].input_position=127;assert!(require_chain(&prompts,&lanes,&ds,&[4861;4]).is_err());ds[1].input_position=128;
    assert!(require_chain(&prompts,&lanes,&ds[..67],&[4861;4]).is_err());
    lanes[3][127]+=1;assert!(require_chain(&prompts,&lanes,&ds,&[4861;4]).is_err());
}
