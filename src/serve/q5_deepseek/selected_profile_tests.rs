use super::*;
fn opts() -> LoadOptions {
    LoadOptions { model_path:"/bounded/mock-selected.gguf".into(),tokenizer_path:None,config_path:None,
        dwq_overlay_path:None,kv_persist_dir:None,kv_persist_budget_bytes:12345,
        glp_path:None,glp_alpha:None,kv_graft_path:None }
}
fn routes()->RuntimeRoutes { RuntimeRoutes {mmap_raw:true,kernel_profile:false,general_profile:false,parallel_encode:false} }
#[test]
fn actual_options_origin_and_unused_budget_are_not_argv_inference() {
    let mut o=opts();assert_eq!(TokenizerOrigin::from_options(&o),TokenizerOrigin::EmbeddedGguf);
    let p=SelectedLoadProfile::derive(&o,Some(262144),routes()).unwrap();
    assert!(!p.persistence_option_present);assert_eq!(p.persistence_budget_bytes,12345);
    assert_eq!(p.effective_context,262144);assert!(!p.runtime_qualified);
    assert!(p.output_limit.starts_with("ExplicitUnknown"));assert!(p.queue_capacity.starts_with("ExplicitUnknown"));
    assert!(p.logical_slots.starts_with("ExplicitUnknown"));assert!(p.loaded_persistor_graft.starts_with("ExplicitUnknown"));
    o.tokenizer_path=Some("/not-opened.json".into());
    assert_eq!(TokenizerOrigin::from_options(&o),TokenizerOrigin::ExternalOverride);
    assert!(SelectedLoadProfile::derive(&o,Some(262144),routes()).is_err());
    assert!(o.tokenizer_path.is_some()); // no silent feature disable.
}
#[test]
fn actual_option_coactivations_refuse_before_loader() {
    for which in 0..7 {
        let mut o=opts();
        match which {
            0=>o.config_path=Some("/not-opened".into()),1=>o.dwq_overlay_path=Some("/not-opened".into()),
            2=>o.kv_persist_dir=Some("/not-created".into()),3=>o.glp_path=Some("/not-opened".into()),
            4=>o.glp_alpha=Some(1.0),5=>o.kv_graft_path=Some("/not-opened".into()),
            _=>o.glp_alpha=Some(f32::NAN),
        }
        assert!(SelectedLoadProfile::derive(&o,Some(262144),routes()).is_err(),"case{which}");
    }
}
#[test]
fn actual_effective_context_and_source_route_must_match_selected_subset() {
    for c in [None,Some(0),Some(131072),Some(262143)] {
        assert!(SelectedLoadProfile::derive(&opts(),c,routes()).is_err());
    }
    for which in 0..4 {
        let mut r=routes();match which {0=>r.mmap_raw=false,1=>r.kernel_profile=true,2=>r.general_profile=true,_=>r.parallel_encode=true}
        assert!(SelectedLoadProfile::derive(&opts(),Some(262144),r).is_err());
    }
    let mut o=opts();o.model_path="relative.gguf".into();
    assert!(SelectedLoadProfile::derive(&o,Some(262144),routes()).is_err());
}
#[test]
fn actual_profile_binding_requires_attached_samefd_context_not_a_caller_record() {
    use crate::serve::q5_deepseek::collector;
    let p=SelectedLoadProfile::derive(&opts(),Some(262144),routes()).unwrap();
    assert!(collector::bind_selected_load_profile(p).is_err()); // no TLS is not a permit.
    let _scope=collector::HeaderScope::enter().unwrap();
    assert!(collector::bind_selected_load_profile(p).is_err()); // no actual retainedFD.
    assert!(collector::checked_health().is_err()); // existing sticky failure.
}
