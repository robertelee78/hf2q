use super::model::Deepseek4Model;
use super::residency_tests::{open_fixture, tiny_config};

#[test]
fn layer0_ffn_rejects_nonproduction_moe_contract() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let state = model.embed_hyper_state(&[2]).unwrap();
    let error = model
        .forward_layer0_ffn_one(&state, 2)
        .expect_err("tiny MoE dimensions must fail closed");
    assert!(error.to_string().contains("4096/256/6/2048/enabled"));
}
