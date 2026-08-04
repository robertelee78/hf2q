use super::model::Deepseek4Model;
use super::residency_tests::{open_fixture, tiny_config};

#[test]
fn q2_k_embeddings_expand_to_four_identical_hc_streams() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let expected_table = gguf
        .load_tensor_f32("token_embd.weight", model.ctx.device())
        .unwrap();
    let expected = expected_table.as_slice::<f32>().unwrap();

    let actual = model.embed_hyper_state(&[2, 0]).unwrap();
    assert_eq!(actual.shape(), &[2, 4, cfg.hidden_size as usize]);
    let actual = actual.as_slice::<f32>().unwrap();
    let hidden = cfg.hidden_size as usize;
    for (token_slot, token_id) in [2usize, 0].into_iter().enumerate() {
        let expected_row = &expected[token_id * hidden..(token_id + 1) * hidden];
        assert!(expected_row.iter().any(|value| *value != 0.0));
        for lane in 0..4 {
            let offset = (token_slot * 4 + lane) * hidden;
            assert_eq!(&actual[offset..offset + hidden], expected_row);
        }
    }
}

#[test]
fn embedding_forward_rejects_empty_input() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    assert!(model.embed_hyper_state(&[]).is_err());
}
