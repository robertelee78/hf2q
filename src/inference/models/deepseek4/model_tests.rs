use super::model::Deepseek4Model;
use super::residency_tests::{open_fixture, tiny_config};

#[test]
fn native_model_load_keeps_weights_and_cache_on_one_device() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = Deepseek4Model::load_from_gguf(&gguf).unwrap();

    assert_eq!(model.cfg, cfg);
    assert!(!model.weights.is_empty());

    let plan = model.cache_plan(128).unwrap();
    let cache = model.allocate_cache(128).unwrap();
    assert_eq!(cache.layers().len(), cfg.num_hidden_layers as usize);
    assert_eq!(cache.resident_bytes(), plan.resident_bytes);
}

#[test]
fn config_only_parses_without_allocating_model_weights() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    assert_eq!(Deepseek4Model::load_config_only(&gguf).unwrap(), cfg);
}
