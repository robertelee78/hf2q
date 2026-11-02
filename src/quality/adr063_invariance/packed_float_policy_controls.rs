//! Actual comparator branches; existing fixture helpers cannot attest model execution.
use super::*;

#[test]
fn packed_floating_u8_policy_is_not_discrete_id_policy() {
    for stage in [Stage::CacheKeyPacked,Stage::CacheValuePacked] {
        let a=Root::new();let b=Root::new();
        let l=trace(&a,stage,Scalar::U8,&[1,2]);let r=trace(&b,stage,Scalar::U8,&[1,3]);
        let unspecified=compare(&a.0,&l,&b.0,&r,&BTreeMap::new());
        assert_eq!(unspecified.disposition,Disposition::ExplicitInconclusiveNumericPolicy);
        assert!(unspecified.first_difference.is_some());assert_eq!(unspecified.compared_bytes,2);
        assert_eq!(unspecified.max_abs,0.0);assert_eq!(unspecified.rms,0.0);
        assert!(!unspecified.captured_extent_qualified);
        let exact=compare(&a.0,&l,&b.0,&r,&BTreeMap::from([(stage,NumericPolicy::Exact)]));
        assert_eq!(exact.disposition,Disposition::DivergedExactInvariant);
        let c=Root::new();let identical=trace(&c,stage,Scalar::U8,&[1,2]);
        assert_eq!(compare(&a.0,&l,&c.0,&identical,&BTreeMap::from([(stage,NumericPolicy::Exact)])).disposition,
            Disposition::ExactEqual);
        let qualified=compare(&a.0,&l,&b.0,&r,&BTreeMap::from([(stage,NumericPolicy::SourceQualified {
            max_abs:1.0,relative_rms:1.0,source:"HOST_CONTROL_NOT_A_CODEBOOK_DECODER".into()})]));
        assert_eq!(qualified.disposition,Disposition::InvalidEvidence);
    }
    for (stage,scalar,bytes) in [(Stage::CacheState,Scalar::U8,vec![1u8]),
        (Stage::RouterIds,Scalar::U32,1u32.to_le_bytes().to_vec()),
        (Stage::SpecCanonicalIds,Scalar::U32,1u32.to_le_bytes().to_vec())] {
        let a=Root::new();let b=Root::new();let mut changed=bytes.clone();changed[0]=2;
        let l=trace(&a,stage,scalar,&bytes);let r=trace(&b,stage,scalar,&changed);
        assert_eq!(compare(&a.0,&l,&b.0,&r,&BTreeMap::new()).disposition,Disposition::DivergedExactInvariant);
    }
}
