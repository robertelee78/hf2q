use super::model::Deepseek4Model;
use super::residency_tests::{open_fixture, tiny_config};
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

/// ds4's 0731 file stores `token_embd` as F16 (ADR-063 Q4-DS4-RACE L5): the
/// gathered rows equal the CPU f16→f32 decode of the same table rows, in all
/// four HC streams. Q2_K/Q8_0 tables keep their own gathers (tests above).
#[test]
fn f16_embeddings_gather_cpu_decoded_rows_into_four_hc_streams() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let vocab = cfg.vocab_size as usize;
    let hidden = cfg.hidden_size as usize;
    let table: Vec<half::f16> = (0..vocab * hidden)
        .map(|index| half::f16::from_f32(((index * 37 % 211) as f32 - 105.0) * 0.0371))
        .collect();
    let mut weight = model
        .ctx
        .device()
        .alloc_buffer(table.len() * 2, DType::F16, vec![vocab, hidden])
        .unwrap();
    weight.as_mut_slice::<u16>().unwrap().copy_from_slice(
        &table
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
    let token_ids = [2u32, 0, 2];
    let arena = model.prepare_embedding_arena(&token_ids).unwrap();
    let (executor, registry) = model.ctx.split();
    let mut session = executor.begin().unwrap();
    Deepseek4Model::encode_embedding_hyper_state(
        &mut session,
        registry,
        executor.device(),
        &weight,
        GgmlType::F16,
        &[vocab, hidden],
        &arena,
        token_ids.len(),
        vocab,
        hidden,
        cfg.hyper_connection_count,
    )
    .unwrap();
    session.finish().unwrap();

    let lanes = cfg.hyper_connection_count as usize;
    let actual = arena.state.as_slice::<f32>().unwrap();
    for (token_slot, &token_id) in token_ids.iter().enumerate() {
        let row = &table[token_id as usize * hidden..(token_id as usize + 1) * hidden];
        let expected: Vec<u32> = row.iter().map(|value| value.to_f32().to_bits()).collect();
        assert!(row.iter().any(|value| value.to_f32() != 0.0));
        for lane in 0..lanes {
            let offset = (token_slot * lanes + lane) * hidden;
            let got: Vec<u32> = actual[offset..offset + hidden]
                .iter()
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(got, expected, "token slot {token_slot} lane {lane}");
        }
    }
}

#[test]
fn integer_token_embeddings_fail_closed() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let vocab = cfg.vocab_size as usize;
    let hidden = cfg.hidden_size as usize;
    let weight = model
        .ctx
        .device()
        .alloc_buffer(vocab * hidden * 4, DType::I32, vec![vocab, hidden])
        .unwrap();
    let arena = model.prepare_embedding_arena(&[0]).unwrap();
    let (executor, registry) = model.ctx.split();
    let mut session = executor.begin().unwrap();
    let error = Deepseek4Model::encode_embedding_hyper_state(
        &mut session,
        registry,
        executor.device(),
        &weight,
        GgmlType::I32,
        &[vocab, hidden],
        &arena,
        1,
        vocab,
        hidden,
        cfg.hyper_connection_count,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("token embeddings"), "{error}");
}
