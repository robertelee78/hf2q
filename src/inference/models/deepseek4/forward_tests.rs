use super::model::Deepseek4Model;
use super::residency_tests::{open_fixture, open_fixture_with_embedding_type, tiny_config};
use mlx_native::{DType, GgmlType};

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
    let error = model
        .embed_hyper_state(&[cfg.vocab_size])
        .expect_err("out-of-vocabulary token must fail before encoding");
    assert!(error.to_string().contains("exceeds vocabulary"));
}

#[test]
fn native_bf16_embeddings_are_gathered_from_the_artifact_representation() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let vocab = cfg.vocab_size as usize;
    let hidden = cfg.hidden_size as usize;
    let values: Vec<half::bf16> = (0..vocab * hidden)
        .map(|index| half::bf16::from_f32(((index % 43) as f32 - 21.0) / 8.0))
        .collect();
    let mut weight = model
        .ctx
        .device()
        .alloc_buffer(
            values.len() * DType::BF16.size_of(),
            DType::BF16,
            vec![vocab, hidden],
        )
        .unwrap();
    weight
        .as_mut_slice::<half::bf16>()
        .unwrap()
        .copy_from_slice(&values);
    let arena = model.prepare_embedding_arena(&[3, 1]).unwrap();
    let (executor, registry) = model.ctx.split();
    let mut session = executor.begin().unwrap();
    Deepseek4Model::encode_embedding_hyper_state(
        &mut session,
        registry,
        executor.device(),
        &weight,
        GgmlType::BF16,
        &[vocab, hidden],
        &arena,
        2,
        vocab,
        hidden,
        cfg.hyper_connection_count,
    )
    .unwrap();
    session.finish().unwrap();

    let actual = arena.state.as_slice::<f32>().unwrap();
    for (token_slot, token_id) in [3usize, 1].into_iter().enumerate() {
        let expected: Vec<f32> = values[token_id * hidden..(token_id + 1) * hidden]
            .iter()
            .map(|value| value.to_f32())
            .collect();
        for lane in 0..cfg.hyper_connection_count as usize {
            let offset = (token_slot * cfg.hyper_connection_count as usize + lane) * hidden;
            assert_eq!(&actual[offset..offset + hidden], expected);
        }
    }
    assert_eq!(weight.dtype(), DType::BF16);
    assert_eq!(weight.data_byte_len(), values.len() * 2);
}

#[test]
fn q5_embeddings_run_directly_from_their_gguf_blocks() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    for (fixture_type, native_type) in [
        (crate::quantize::ggml_quants::GgmlType::Q5_0, GgmlType::Q5_0),
        (crate::quantize::ggml_quants::GgmlType::Q5_K, GgmlType::Q5_K),
    ] {
        let cfg = tiny_config();
        let (_directory, gguf) =
            open_fixture_with_embedding_type(&cfg, false, false, Some(fixture_type));
        let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
        let embedding = model.weights.raw_matrix_ref("token_embd.weight").unwrap();
        assert_eq!(embedding.ggml_type, native_type);
        assert_eq!(embedding.buffer.dtype(), DType::U8);
        assert!(embedding.buffer.is_file_backed());

        let expected_table = gguf
            .load_tensor_f32("token_embd.weight", model.ctx.device())
            .unwrap();
        let expected_table = expected_table.as_slice::<f32>().unwrap();
        let actual = model.embed_hyper_state(&[2, 0]).unwrap();
        let actual = actual.as_slice::<f32>().unwrap();
        let hidden = cfg.hidden_size as usize;
        for (token_slot, token_id) in [2usize, 0].into_iter().enumerate() {
            let expected = &expected_table[token_id * hidden..(token_id + 1) * hidden];
            assert!(expected.iter().any(|value| *value != 0.0));
            for lane in 0..cfg.hyper_connection_count as usize {
                let offset = (token_slot * cfg.hyper_connection_count as usize + lane) * hidden;
                for (column, (actual, expected)) in actual[offset..offset + hidden]
                    .iter()
                    .zip(expected)
                    .enumerate()
                {
                    assert!(
                        (actual - expected).abs() <= 1e-6,
                        "{native_type:?} embedding column {column}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn q8_0_embeddings_expand_to_four_identical_hc_streams() {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let vocab = cfg.vocab_size as usize;
    let hidden = cfg.hidden_size as usize;
    let blocks_per_row = hidden / BLOCK_VALUES;
    let mut bytes = vec![0u8; vocab * blocks_per_row * BLOCK_BYTES];
    for row in 0..vocab {
        for block in 0..blocks_per_row {
            let offset = (row * blocks_per_row + block) * BLOCK_BYTES;
            bytes[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            for column in 0..BLOCK_VALUES {
                bytes[offset + 2 + column] =
                    ((row as i16 + block as i16 + column as i16) % 31 - 15) as i8 as u8;
            }
        }
    }
    let mut weight = model
        .ctx
        .device()
        .alloc_buffer(bytes.len(), DType::U8, vec![vocab, hidden])
        .unwrap();
    weight.as_mut_slice::<u8>().unwrap().copy_from_slice(&bytes);
    let arena = model.prepare_embedding_arena(&[2, 0]).unwrap();
    let (executor, registry) = model.ctx.split();
    let mut session = executor.begin().unwrap();
    Deepseek4Model::encode_embedding_hyper_state(
        &mut session,
        registry,
        executor.device(),
        &weight,
        GgmlType::Q8_0,
        &[vocab, hidden],
        &arena,
        2,
        vocab,
        hidden,
        cfg.hyper_connection_count,
    )
    .unwrap();
    session.finish().unwrap();

    let actual = arena.state.as_slice::<f32>().unwrap();
    for (token_slot, token_id) in [2usize, 0].into_iter().enumerate() {
        for lane in 0..cfg.hyper_connection_count as usize {
            let offset = (token_slot * cfg.hyper_connection_count as usize + lane) * hidden;
            for column in 0..hidden {
                let expected =
                    ((token_id + column / BLOCK_VALUES + column % BLOCK_VALUES) % 31) as i32 - 15;
                assert_eq!(actual[offset + column], expected as f32);
            }
        }
    }
}
