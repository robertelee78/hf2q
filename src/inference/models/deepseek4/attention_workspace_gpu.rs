//! Actual common prelude/RoPE/epilogue comparison, not whole attention/model.
use super::*;
use crate::inference::models::deepseek4::{
    compressed_attention_common::{
        encode_compressed_attention_epilogue, encode_compressed_attention_prelude,
        CompressedAttentionCoreArena,
    },
    rope::yarn_frequencies,
};
use mlx_native::ops::{
    deepseek_tail_rope::{dispatch_deepseek_tail_rope_f32_to_bf16, DeepSeekTailRopeParams},
    quantized_matmul_ggml::GgmlType,
};
#[path = "attention_workspace_fixture.rs"]
mod fixture;
use fixture::*;
const LIMIT: usize = 256 * 1024 * 1024;
fn planned(rows: usize) -> usize {
    let hc = 16384;
    let h = 32768;
    let hidden = 4096;
    let rank = 1024;
    let out = 8192;
    let weights = 114080108usize;
    let core = rows
        * (hc + 24 + 4 + 4 + 16 + hidden * 2 + rank * 2 + h * 3 + 512 * 2 + out + hidden + hc)
        * 4
        + rows * h * 2
        + 32
        + if rows > 1 { rows * out * 4 } else { 0 };
    let saving = rows * (hc + h) * 4;
    let inputs =
        rows * hc * 4 + rows * h * 2 + rows * 4 + 32 * 4 + 2 * rows * h * 2 + 4 * 2 * GUARD * 4;
    weights + 2 * core - saving + inputs + 32 * 1024 * 1024
}
fn core_bytes(a: &CompressedAttentionCoreArena) -> usize {
    let mut roots = std::collections::BTreeMap::new();
    for b in [
        &a.state_norm,
        &a.mixes,
        &a.pre,
        &a.post,
        &a.comb,
        &a.attn_input,
        &a.attn_norm,
        &a.q_a,
        &a.q_a_norm,
        &a.q,
        &a.q_norm,
        &a.kv,
        &a.kv_norm,
        &a.attention_unrotated,
        &a.attention_f32,
        &a.output_a,
        &a.output_b,
        &a.output_state,
        &a.hc_params,
        &a.hidden_params,
        &a.rank_params,
        &a.head_params,
    ] {
        roots.insert(b.contents_ptr() as usize, b.byte_len());
    }
    roots.values().sum::<usize>()
        + a.grouped_output
            .as_ref()
            .map_or(0, |a| a.parent_bytes_for_test())
}
fn poison(a: &mut CompressedAttentionCoreArena, parents: &mut [MlxBuffer]) {
    for b in parents {
        b.as_logical_mut_slice::<u32>().unwrap().fill(HOSTILE);
    }
    for b in [
        &mut a.q_norm,
        &mut a.output_a,
        &mut a.output_b,
        &mut a.output_state,
    ] {
        b.as_logical_mut_slice::<f32>()
            .unwrap()
            .fill(f32::from_bits(HOSTILE));
    }
    a.attention_unrotated
        .as_logical_mut_slice::<u16>()
        .unwrap()
        .fill(0x7fc1);
}
#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &mut crate::serve::gpu::GpuContext,
    d: &MlxDevice,
    c: &super::super::Deepseek4Config,
    w: &Weights,
    state: &MlxBuffer,
    attention: &MlxBuffer,
    positions: &MlxBuffer,
    freq: &MlxBuffer,
    q_rope: &MlxBuffer,
    a: &CompressedAttentionCoreArena,
    rows: usize,
    recorded: bool,
) {
    let (exec, reg) = ctx.split();
    let mut s = if recorded {
        exec.begin_recorded()
    } else {
        exec.begin()
    }
    .unwrap();
    encode_compressed_attention_prelude(&mut s, reg, d, c, state, &w.refs(), a, rows).unwrap();
    s.barrier_between(&[&a.q_norm, positions, freq], &[q_rope]);
    dispatch_deepseek_tail_rope_f32_to_bf16(
        s.encoder_mut(),
        reg,
        d,
        &a.q_norm,
        positions,
        freq,
        q_rope,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: rows as u32,
            heads: 64,
            head_dim: 512,
            rope_dim: 64,
            inverse: 0,
        },
    )
    .unwrap();
    // Fixed BF16 attention fixture: no CPU attention implementation or claim
    // to test ratio-specific compressors/indexer/cache attention numerics.
    encode_compressed_attention_epilogue(
        &mut s,
        reg,
        d,
        c,
        state,
        &w.refs(),
        a,
        attention,
        positions,
        freq,
        rows,
    )
    .unwrap();
    if recorded {
        let (reordered, barriers) = s
            .finish_with_reorder()
            .expect("actual recorded/reorder checked completion");
        assert!(
            barriers > 0,
            "recorded dependency chain must emit actual barriers, not an empty capture"
        );
        println!("DS_ATTENTION_WORKSPACE_REORDER reordered={reordered} barriers={barriers}");
    } else {
        s.finish().expect("actual direct checked completion");
    }
}
fn parity(a: &MlxBuffer, b: &MlxBuffer) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    let x = a.as_logical_slice::<u8>().unwrap();
    let y = b.as_logical_slice::<u8>().unwrap();
    assert_eq!(x.len(), y.len());
    assert!(
        x.iter().zip(y).all(|(x, y)| x == y),
        "old/shared full bits differ"
    );
    match a.dtype() {
        DType::F32 => {
            let v = a.as_logical_slice::<f32>().unwrap();
            assert!(v.iter().all(|x| x.is_finite()));
            assert!(v.iter().any(|x| *x != 0.));
        }
        DType::BF16 => {
            let v = a.as_logical_slice::<u16>().unwrap();
            assert!(v.iter().all(|&x| half::bf16::from_bits(x).is_finite()));
            assert!(v.iter().any(|&x| half::bf16::from_bits(x).to_f32() != 0.));
        }
        _ => panic!("unexpected comparison scalar"),
    }
}
#[test]
fn common_adapter_fixture_bound_is_finite_before_device_creation() {
    assert_eq!(planned(65), 245111984);
    assert!(planned(65) <= LIMIT);
    assert!(planned(1) < planned(65));
}
#[test]
#[ignore = "root-supervised tiny actual Metal control <=256MiB requested; no model/latency claim"]
fn actual_common_adapters_separate_vs_shared_direct_and_reorder_current_main_port() {
    assert_eq!(planned(65), 245111984);
    assert!(planned(65) <= LIMIT);
    assert!(
        !std::env::vars().any(|(k, _)| k.starts_with("HF2Q_")
            || k.starts_with("MLX_")
            || k.starts_with("MTL_")
            || k.starts_with("METAL_")),
        "ambient experiment activation"
    );
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = crate::serve::gpu::GpuContext::new().expect("actual Metal required, no skip");
    let d = ctx.device().clone();
    {
        let p = Plan::new(1, 4, 2, 2).unwrap();
        let wrong = alloc(&d, DType::U8, vec![16], "wrong workspace dtype").unwrap();
        assert!(AttentionWorkspace::from_root(wrong, p).is_err());
        let oversized = alloc(&d, DType::F32, vec![5], "wrong workspace extent").unwrap();
        assert!(AttentionWorkspace::from_root(oversized, p).is_err());
    }
    {
        let mut e = d.command_encoder().unwrap();
        e.commit_and_wait()
            .expect("checked rejected-fixture removal");
    }
    let w = Weights::new(&d);
    assert_eq!(w.bytes(), 114080108);
    let source = option_env!("HF2Q_BUILD_GIT_SHA").expect("source-bound actual control");
    assert!(source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()));
    println!("DS_ATTENTION_WORKSPACE_BEGIN requested_upper={} limit={} native={} source={} device_registry={} scope=common_adapters_not_full_attention",planned(65),LIMIT,env!("HF2Q_MLX_NATIVE_VERSION"),source,d.registry_id());
    let mut cases = 0;
    for ratio in [0, 4, 128] {
        for rows in [1usize, 65] {
            for recorded in [false, true] {
                {
                    let c = cfg(ratio);
                    let p = Plan::new(rows, 16384, 64, 512).unwrap();
                    let mut op = Vec::new();
                    let mut np = Vec::new();
                    let old = workspace(&d, p, false, &mut op);
                    let new = workspace(&d, p, true, &mut np);
                    assert_ne!(old.state_norm.contents_ptr(), old.q.contents_ptr());
                    assert_ne!(old.q.contents_ptr(), old.attention_f32.contents_ptr());
                    assert_eq!(new.state_norm.contents_ptr(), new.q.contents_ptr());
                    assert_eq!(new.q.contents_ptr(), new.attention_f32.contents_ptr());
                    let mut a = CompressedAttentionCoreArena::with_workspace(
                        &d,
                        &c,
                        rows,
                        GgmlType::Q8_0,
                        old,
                    )
                    .unwrap();
                    let mut b = CompressedAttentionCoreArena::with_workspace(
                        &d,
                        &c,
                        rows,
                        GgmlType::Q8_0,
                        new,
                    )
                    .unwrap();
                    assert_ne!(
                        b.q_norm.contents_ptr(),
                        b.q.contents_ptr(),
                        "still-live norm must stay distinct"
                    );
                    let mut state =
                        alloc(&d, DType::F32, vec![rows, 4, 4096], "common state fixture").unwrap();
                    for (i, x) in state
                        .as_logical_mut_slice::<f32>()
                        .unwrap()
                        .iter_mut()
                        .enumerate()
                    {
                        *x = ((i % 31) as f32 - 15.) * 0.03125;
                    }
                    let mut attention = alloc(
                        &d,
                        DType::BF16,
                        vec![1, rows, 64, 512],
                        "fixed attention fixture",
                    )
                    .unwrap();
                    for (i, x) in attention
                        .as_logical_mut_slice::<u16>()
                        .unwrap()
                        .iter_mut()
                        .enumerate()
                    {
                        *x = half::bf16::from_f32(((i % 23) as f32 - 11.) * 0.0625).to_bits();
                    }
                    let mut pos = alloc(&d, DType::U32, vec![rows], "positions fixture").unwrap();
                    for (i, x) in pos
                        .as_logical_mut_slice::<u32>()
                        .unwrap()
                        .iter_mut()
                        .enumerate()
                    {
                        *x = i as u32;
                    }
                    let frequencies = yarn_frequencies(
                        64,
                        65536,
                        if ratio == 0 {
                            c.rope_theta
                        } else {
                            c.compress_rope_theta
                        },
                        16.,
                        32.,
                        1.,
                    )
                    .unwrap();
                    let mut freq =
                        alloc(&d, DType::F32, vec![frequencies.len()], "RoPE fixture").unwrap();
                    freq.as_logical_mut_slice::<f32>()
                        .unwrap()
                        .copy_from_slice(&frequencies);
                    let aq = alloc(&d, DType::BF16, vec![1, rows, 64, 512], "old rotated query")
                        .unwrap();
                    let bq = alloc(
                        &d,
                        DType::BF16,
                        vec![1, rows, 64, 512],
                        "shared rotated query",
                    )
                    .unwrap();
                    let actual = w.bytes()
                        + core_bytes(&a)
                        + core_bytes(&b)
                        + [&state, &attention, &pos, &freq, &aq, &bq]
                            .iter()
                            .map(|b| b.byte_len())
                            .sum::<usize>();
                    assert!(
                        actual + 32 * 1024 * 1024 <= LIMIT,
                        "actual parent bytes exceed declared numeric fixture"
                    );
                    for _round in 0..2 {
                        poison(&mut a, &mut op);
                        poison(&mut b, &mut np);
                        run(
                            &mut ctx, &d, &c, &w, &state, &attention, &pos, &freq, &aq, &a, rows,
                            recorded,
                        );
                        run(
                            &mut ctx, &d, &c, &w, &state, &attention, &pos, &freq, &bq, &b, rows,
                            recorded,
                        );
                        for (x, y) in [
                            (&a.q_norm, &b.q_norm),
                            (&a.output_a, &b.output_a),
                            (&a.output_b, &b.output_b),
                            (&a.output_state, &b.output_state),
                            (&a.attention_f32, &b.attention_f32),
                            (&aq, &bq),
                        ] {
                            parity(x, y);
                        }
                        guards(&op);
                        guards(&np);
                    }
                    println!("DS_ATTENTION_WORKSPACE_CASE rows={rows} ratio_profile={ratio} recorded={recorded} rounds=2 parent_bytes={actual} exact_bits=true guards=true scope=actual_common_prelude_epilogue_only");
                }
                // Test fixture retirement only, not a production synchronization change.
                // Commit staged removals before allocating the next case's parents.
                {
                    let mut e = d.command_encoder().unwrap();
                    e.commit_and_wait()
                        .expect("checked fixture membership retirement");
                }
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 12);
    println!("DS_ATTENTION_WORKSPACE_COMPLETE cases=12 reuses=2 bounds_refusals=2 exact_bits=true model_loaded=false physical_fit=false speed_qualified=false");
}
