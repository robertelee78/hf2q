use mlx_native::DType;

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

#[test]
fn native_output_head_produces_finite_vocab_logits_and_rejects_shape_drift() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let mut state = model
        .ctx
        .device()
        .alloc_buffer(
            4 * cfg.hidden_size as usize * DType::F32.size_of(),
            DType::F32,
            vec![1, 4, cfg.hidden_size as usize],
        )
        .unwrap();
    for (index, value) in state.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *value = (index as f32 * 0.013).sin();
    }
    let logits = model.forward_logits(&state).unwrap();
    assert_eq!(logits.shape(), &[1, cfg.vocab_size as usize]);
    assert!(logits
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
    assert_eq!(model.greedy_token(&logits).unwrap(), 0);

    let wrong = model
        .ctx
        .device()
        .alloc_buffer(8 * 4, DType::F32, vec![1, 2, 4])
        .unwrap();
    assert!(model.forward_logits(&wrong).is_err());
    assert!(model.greedy_token(&wrong).is_err());
}
