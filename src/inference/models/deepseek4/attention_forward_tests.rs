use super::model::Deepseek4Model;
use super::residency_tests::{open_fixture, tiny_config};

#[test]
fn failed_layer0_attention_does_not_publish_cache_state() {
    let mut cfg = tiny_config();
    cfg.compress_ratios[0] = 0;
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let mut cache = model.allocate_cache(8).unwrap();

    let _error = model
        .forward_layer0_attention_one(2, &mut cache)
        .expect_err("tiny non-production attention shape must fail closed");
    assert_eq!(
        cache.position(),
        0,
        "failed command buffer must not publish cache state"
    );
}
