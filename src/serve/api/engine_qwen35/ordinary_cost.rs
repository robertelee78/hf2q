//! Cost observations, not latency or a claimed counterfactual scalar timing.
use std::time::{Duration,Instant};
#[derive(Clone,Copy)]
pub(in crate::serve::api) enum OrdinaryCost {
    OriginalScalar(Instant),
    MeasuredTarget { raw:Duration,width:u32,own_prepare:Duration },
}
fn measured_service(raw:Duration,width:u32,prepare:Duration,tail:Duration)->(Duration,Duration) {
    let allocated=raw.checked_div(width).expect("validated nonzero width");
    (allocated,allocated.saturating_add(prepare).saturating_add(tail))
}
impl OrdinaryCost {
    pub(in crate::serve::api) fn measured(raw:Duration,width:usize,own_prepare:Duration)->anyhow::Result<Self> {
        anyhow::ensure!((1..=4).contains(&width)&&!raw.is_zero(),"measured ordinary target duration/width");
        Ok(Self::MeasuredTarget {raw,width:width as u32,own_prepare})
    }
    pub(in crate::serve::api) fn complete(self,own_tail_started:Instant)->Duration {
        match self {
            Self::OriginalScalar(started)=>started.elapsed(),
            Self::MeasuredTarget {raw,width,own_prepare}=>{
                let tail=own_tail_started.elapsed();
                let (allocated,observed)=measured_service(raw,width,own_prepare,tail);
                tracing::info!(raw_target_ns=raw.as_nanos() as u64,width,
                    allocated_target_ns=allocated.as_nanos() as u64,
                    own_prepare_ns=own_prepare.as_nanos() as u64,own_tail_ns=tail.as_nanos() as u64,
                    observed_service_ns=observed.as_nanos() as u64,
                    basis="equal share actual target + own tail; excludes other-owner queue wait; NOT scalar counterfactual/latency",
                    "Qwen ordinary measured service cost");
                observed
            }
        }
    }
}
#[cfg(test)]mod controls {
    use super::*;
    #[test]fn actual_shared_cost_keeps_raw_target_and_width_not_full_charge_per_owner() {
        let cost=OrdinaryCost::measured(Duration::from_millis(400),4,Duration::from_millis(2)).unwrap();
        let OrdinaryCost::MeasuredTarget {raw,width,own_prepare}=cost else {panic!("measured")};
        assert_eq!(raw,Duration::from_millis(400));assert_eq!(width,4);
        let (share,total)=measured_service(raw,width,own_prepare,Duration::from_millis(3));
        assert_eq!(share,Duration::from_millis(100));
        assert_eq!(total,Duration::from_millis(105)); // no other-owner wait term
        assert_eq!(measured_service(raw,1,own_prepare,Duration::from_millis(3)).1,
            Duration::from_millis(405));
        for width in [0,5] {assert!(OrdinaryCost::measured(raw,width,own_prepare).is_err());}
        assert!(OrdinaryCost::measured(Duration::ZERO,4,Duration::ZERO).is_err());
    }
}
