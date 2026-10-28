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
