use super::*;
use crate::inference::models::qwen35::encoder_stage::{settle_submitted, SubmittedCommands};
use crate::inference::models::qwen35::kv_cache::TqFullAttnKvBuffers;
use crate::inference::models::qwen35::{FaPrefillArena, FaProjectionsArena};

#[test]
fn tq_cold_fused_layer_matches_the_persisted_cache_bridge() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        eprintln!("skipping TQ fused attention parity: no Metal device");
        return;
    };
    let _scratch = crate::inference::models::qwen35::prepared_scratch::ExecutionScratch::test_scope(&device);
    let _prepared = crate::inference::models::qwen35::prefill_prepared::PrefillScope::forced(false);
    let _route = crate::inference::models::qwen35::resume_attn_fast::ForceResumeAttnRoute::new(true);
    let mut registry = KernelRegistry::new();
    mlx_native::ops::flash_attn_prefill::register(&mut registry);
    let (hidden, heads, kv_heads, dim) = (256u32, 4u32, 2u32, 256u32);
    let mut seed = 0xc23fa;
    let weights = layer_weights_f32(
        hidden as usize, heads as usize, kv_heads as usize, dim as usize, &mut seed, &device,
    );
    for rows in [16u32, 33, 97] {
        let capacity = rows + 64;
        let input = upload_f32(&mk_rand(&mut seed, (rows * hidden) as usize, 0.5), &device).unwrap();
        let positions = upload_positions(rows as usize, 0, &device);
        let mut outputs = Vec::with_capacity(2);
        for fused in [false, true] {
            let packed = || device.alloc_buffer(
                (kv_heads * capacity * dim) as usize,
                DType::U8,
                vec![1, kv_heads as usize, capacity as usize, dim as usize],
            ).unwrap();
            let norms = || device.alloc_buffer(
                (kv_heads * capacity * 4) as usize,
                DType::F32,
                vec![1, kv_heads as usize, capacity as usize, 1],
            ).unwrap();
            let mut slot = FullAttnKvSlot {
                k: None,
                v: None,
                current_len: vec![0],
                tq: Some(TqFullAttnKvBuffers {
                    k_packed: packed(),
                    k_norms: norms(),
                    v_packed: packed(),
                    v_norms: norms(),
                    norms_per_pos: 1,
                }),
            };
            let mut prefill = fused.then(|| {
                FaPrefillArena::new(&device, rows, heads, kv_heads, dim).unwrap()
            });
            let mut projections = FaProjectionsArena::new(
                &device, rows, hidden, heads, kv_heads, dim, 1e-6,
            ).unwrap();
            let _submitted = SubmittedCommands::open(true);
            let mut session = device.encoder_session().unwrap();
            outputs.push(build_gated_attn_layer(
                &device, &mut registry, &input, &positions, &weights, Some(&mut slot),
                capacity, rows, hidden, heads, kv_heads, dim, 64, 1e7,
                [11, 11, 10, 0], 1e-6, prefill.as_mut(), Some(&mut projections),
                None, session.as_mut(), SlotId(0),
            ).unwrap());
            if let Some(session) = session.as_mut() {
                LayerEncoder::Sessioned(session)
                    .commit_and_wait_labeled_terminal("test.tq_cold_fused.terminal").unwrap();
            }
            settle_submitted().unwrap();
            device.command_encoder().unwrap().commit_and_wait().unwrap();
            assert_eq!(slot.current_len[0], rows);
        }
        let expected = outputs[0].as_slice::<f32>().unwrap();
        let actual = outputs[1].as_slice::<f32>().unwrap();
        assert!(expected.iter().any(|&value| value != 0.0));
        assert_eq!(actual.len(), (rows * hidden) as usize);
        let mismatch = actual.iter().zip(expected)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(mismatch, None, "fused cold TQ layer differs from cache bridge at rows={rows}");
    }
}
