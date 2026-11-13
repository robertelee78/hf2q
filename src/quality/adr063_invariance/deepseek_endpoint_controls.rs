//! Actual producer scope guards, no model/header/device access.
use super::*;
#[test]
fn ordinary_producer_cannot_relabel_gate_rejection_as_owning_cancellation() {
    for t in Trajectory::ALL {
        assert_eq!(require_ordinary(t).is_ok(),!matches!(t,Trajectory::T5GenuineSpeculation|Trajectory::T6CancelRestore));
    }
    assert_eq!(boundaries(Trajectory::T0ColdSerial),vec![(128,false),(144,false)]);
    assert_eq!(boundaries(Trajectory::T3Chunk64),vec![(128,false),(144,false),(64,false)]);
    assert_eq!(boundaries(Trajectory::T1Resume64),vec![(128,false),(144,false),(64,false),(64,true)]);
    let e=ExpectedTensor {at:coordinate(0,0,Stage::Logits,1),scalar:Scalar::F32,shape:vec![1,129280]};
    assert_eq!(payload(&[e]).unwrap(),517120);
    let bad=ExpectedTensor {at:coordinate(0,0,Stage::Logits,1),scalar:Scalar::F32,shape:vec![usize::MAX,usize::MAX]};
    assert!(payload(&[bad]).is_err());
}

/// DeepSeek Q1 packets declare the three default-on routes; a packet without
/// the fields means the routes of every retained run (Stock / Off / PerRow,
/// the three kill switches), and an undeclared default route is refused.
#[test]
fn q1_deepseek_packet_routes_default_to_the_retained_routes_and_refuse_undeclared_ones() {
    let retained=(MatvecKernels::default(),TailRowsStep::default(),AttnCores::default());
    assert_eq!(retained,(MatvecKernels::Stock,TailRowsStep::Off,AttnCores::PerRow));
    let shipped=(MatvecKernels::RowExact,TailRowsStep::On,AttnCores::Batched);
    assert_eq!(require_declared(retained,retained).unwrap(),
        serde_json::json!({"matvec_kernels":"Stock","tail_rows_step":"Off","attn_cores":"PerRow"}));
    assert_eq!(require_declared(shipped,shipped).unwrap(),
        serde_json::json!({"matvec_kernels":"RowExact","tail_rows_step":"On","attn_cores":"Batched"}));
    assert!(require_declared(retained,shipped).is_err());
    assert!(require_declared(shipped,retained).is_err());
    assert!(require_declared((MatvecKernels::RowExact,TailRowsStep::Off,AttnCores::PerRow),retained).is_err());
    // The harness spellings are the model's audit spellings.
    for (wire,value) in [("Stock",MatvecKernels::Stock),("RowExact",MatvecKernels::RowExact)] {
        assert_eq!(serde_json::from_value::<MatvecKernels>(serde_json::json!(wire)).unwrap(),value);
    }
    assert_eq!(format!("{:?}",crate::inference::models::deepseek4::residency::MatvecKernels::RowExact),"RowExact");
    assert!(serde_json::from_value::<TailRowsStep>(serde_json::json!("on")).is_err());
}
