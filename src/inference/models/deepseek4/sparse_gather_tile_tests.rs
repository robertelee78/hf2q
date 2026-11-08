//! Shared production adapter comparison. Only the ignored control touches Metal.
use super::*;

#[test]
fn sparse_gather_tile_actual_plan_bounds_and_real_width_savings() {
    assert_eq!(SPARSE_PREFILL_QUERY_TILE, 64);
    for (width, old_bytes, new_bytes) in [
        (2176, 570425344, 142606336),
        (2112, 553648128, 138412032),
        (640, 167772160, 41943040),
    ] {
        let old = SparseGatherPlan::new(2048, 64, 512, 65536, width, 256).unwrap();
        let new =
            SparseGatherPlan::new(2048, 64, 512, 65536, width, SPARSE_PREFILL_QUERY_TILE).unwrap();
        assert_eq!(old.gathered_bytes, old_bytes);
        assert_eq!(new.gathered_bytes, new_bytes);
        if width >= 2112 {
            assert_eq!(old.gathered_bytes.next_power_of_two(), 1073741824);
            assert_eq!(new.gathered_bytes.next_power_of_two(), 268435456);
        }
    }
    for rows in [63, 64, 65, 255, 256, 257] {
        let p = SparseGatherPlan::new(rows, 64, 512, rows + 26, 17, 64).unwrap();
        let mut visited = vec![0u8; rows];
        // Same tile origin/count expressions consumed by the production loop.
        for start in (0..rows).step_by(p.query_tile) {
            let count = (rows - start).min(p.query_tile);
            assert!(count <= p.tile_rows());
            for seen in &mut visited[start..start + count] {
                *seen += 1;
            }
        }
        assert!(visited.iter().all(|&n| n == 1));
    }
}

#[test]
fn sparse_gather_tile_invalid_plan_and_call_refuse() {
    for args in [
        (0, 64, 512, 64, 17, 64),
        (33, 64, 512, 0, 17, 64),
        (33, 64, 512, 64, 0, 64),
        (33, 64, 512, 64, 17, 0),
        (33, 64, 512, 64, 17, 128),
        (33, 63, 512, 64, 17, 64),
        (33, 64, 511, 64, 17, 64),
        (usize::MAX, 64, 512, 64, 17, 64),
        (
            u32::MAX as usize,
            u32::MAX as usize,
            u32::MAX as usize,
            64,
            17,
            64,
        ),
    ] {
        assert!(SparseGatherPlan::new(args.0, args.1, args.2, args.3, args.4, args.5).is_err());
    }
    let p = SparseGatherPlan::new(257, 64, 512, 283, 17, 64).unwrap();
    assert!(p.check_call(257, 64, 512, 283, 17).is_ok());
    assert!(p.check_call(258, 64, 512, 283, 17).is_err());
    assert!(p.check_call(257, 64, 512, 283, 18).is_err());
}

fn arena_view(
    parent: &DeepseekSparsePrefillFlashArena,
    rows: usize,
) -> DeepseekSparsePrefillFlashArena {
    let p = SparseGatherPlan::new(rows, 64, 512, rows + 26, 17, parent.plan.query_tile).unwrap();
    DeepseekSparsePrefillFlashArena {
        plan: p,
        kv: parent
            .kv
            .slice_view(0, p.kv_len * 512)
            .with_shape(vec![1, 1, p.kv_len, 512])
            .unwrap(),
        gathered_kv: parent
            .gathered_kv
            .slice_view(0, p.gathered_bytes / 2)
            .with_shape(vec![1, p.tile_rows(), 17, 512])
            .unwrap(),
        mask: parent
            .mask
            .slice_view(0, p.mask_bytes / 2)
            .with_shape(vec![p.tile_rows(), 1, 17])
            .unwrap(),
        invalid_global: parent
            .invalid_global
            .slice_view(0, rows)
            .with_shape(vec![1, rows])
            .unwrap(),
        invalid_heads: parent
            .invalid_heads
            .slice_view(0, rows * 64)
            .with_shape(vec![1, rows, 64])
            .unwrap(),
    }
}

#[test]
#[ignore = "separate supervised tiny Metal control; fixed roots below128MiB, no model"]
fn sparse_gather_tile_actual_old256_new64_exact_bits_flags_and_ragged_offsets() {
    const R: usize = 257;
    const H: usize = 64;
    const D: usize = 512;
    const K: usize = 17;
    const PREFIX: usize = 7;
    const COMP: usize = 19;
    let old_plan = SparseGatherPlan::new(R, H, D, R + PREFIX + COMP, K, 256).unwrap();
    let new_plan = SparseGatherPlan::new(R, H, D, R + PREFIX + COMP, K, 64).unwrap();
    let arena_bytes = |p: SparseGatherPlan| {
        2 * p.kv_len * D + p.gathered_bytes + p.mask_bytes + 4 * R + 4 * R * H
    };
    let declared = arena_bytes(old_plan)
        + arena_bytes(new_plan)
        + 2 * R * H * D
        + 2 * R * D
        + 2 * PREFIX * D
        + 2 * COMP * D
        + 4 * H
        + 4 * R * K
        + 2 * 2 * (R + 2) * H * D
        + 65536;
    assert!(
        declared < 128 * 1024 * 1024,
        "fixed numeric root/host payload checked BEFORE device creation"
    );
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = crate::serve::gpu::GpuContext::new().expect("actual Metal required; never skip");
    let device = ctx.device().clone();
    let make = |dtype, shape, label| alloc(&device, dtype, shape, label).unwrap();
    let mut query = make(DType::BF16, vec![R, H, D], "tile test query");
    let mut raw = make(DType::BF16, vec![R, 1, D], "tile test current KV");
    let mut prefix = make(DType::BF16, vec![PREFIX, 1, D], "tile test prefix KV");
    let mut compressed = make(DType::BF16, vec![COMP, 1, D], "tile test compressed KV");
    for (j, buffer) in [&mut query, &mut raw, &mut prefix, &mut compressed]
        .into_iter()
        .enumerate()
    {
        for (i, v) in buffer
            .as_logical_mut_slice::<u16>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *v = half::bf16::from_f32((((i * 7 + j * 11) % 61) as f32 - 30.0) * 0.015625).to_bits();
        }
    }
    let mut sinks = make(DType::F32, vec![H], "tile test sinks");
    for (i, v) in sinks
        .as_logical_mut_slice::<f32>()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *v = (i % 5) as f32 * 0.125;
    }
    let mut indices = make(DType::I32, vec![1, R, K], "tile test indices");
    let mut old_output = make(DType::BF16, vec![R + 2, H, D], "old output plus guards");
    let mut new_output = make(DType::BF16, vec![R + 2, H, D], "new output plus guards");
    let mut old_arena =
        DeepseekSparsePrefillFlashArena::new_with_tile(&device, R, H, D, R + PREFIX + COMP, K, 256)
            .unwrap();
    let mut new_arena =
        DeepseekSparsePrefillFlashArena::new(&device, R, H, D, R + PREFIX + COMP, K).unwrap();
    assert_eq!(new_arena.plan.query_tile, 64);
    const SENTINEL: u16 = 0x40e0; // BF16 seven; guard rows must never be written.
    for (rows, invalid) in [
        (63, false),
        (64, false),
        (65, false),
        (255, false),
        (256, false),
        (257, false),
        (65, true),
    ] {
        println!("DS_SPARSE_GATHER_TILE_BEGIN rows={rows} invalid={invalid}");
        old_output
            .as_logical_mut_slice::<u16>()
            .unwrap()
            .fill(SENTINEL);
        new_output
            .as_logical_mut_slice::<u16>()
            .unwrap()
            .fill(SENTINEL);
        for arena in [&mut old_arena, &mut new_arena] {
            arena
                .invalid_global
                .as_logical_mut_slice::<u32>()
                .unwrap()
                .fill(0);
            arena
                .invalid_heads
                .as_logical_mut_slice::<u32>()
                .unwrap()
                .fill(0);
        }
        for (q, row) in indices
            .as_logical_mut_slice::<i32>()
            .unwrap()
            .chunks_exact_mut(K)
            .take(rows)
            .enumerate()
        {
            for (i, v) in row.iter_mut().enumerate() {
                *v = match i % 5 {
                    0 => -1,
                    1 => 0,
                    2 => (PREFIX + q) as i32,
                    _ => (PREFIX + rows + i % COMP) as i32,
                };
            }
            if invalid && q == 64 {
                row[1] = (PREFIX + rows + COMP + 1) as i32;
            }
        }
        let q = query
            .slice_view(0, rows * H * D)
            .with_shape(vec![rows, H, D])
            .unwrap();
        let kv = raw
            .slice_view(0, rows * D)
            .with_shape(vec![rows, 1, D])
            .unwrap();
        let ix = indices
            .slice_view(0, rows * K)
            .with_shape(vec![1, rows, K])
            .unwrap();
        let old = arena_view(&old_arena, rows);
        let new = arena_view(&new_arena, rows);
        let old_view = old_output
            .slice_view((H * D * 2) as u64, rows * H * D)
            .with_shape(vec![rows, H, D])
            .unwrap();
        let new_view = new_output
            .slice_view((H * D * 2) as u64, rows * H * D)
            .with_shape(vec![rows, H, D])
            .unwrap();
        for (arena, output) in [(&old, &old_view), (&new, &new_view)] {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            encode_deepseek_sparse_flash_prefill(
                &mut session,
                registry,
                &device,
                &q,
                Some(&prefix),
                &kv,
                &compressed,
                &sinks,
                &ix,
                output,
                arena,
                rows,
                PREFIX,
                PREFIX + rows,
                PREFIX + rows + COMP,
                K,
                H,
                D,
                1.0 / (D as f32).sqrt(),
            )
            .unwrap();
            session
                .finish()
                .expect("actual sparse prefill command must complete");
        }
        let a = old_view.as_logical_slice::<u16>().unwrap();
        let b = new_view.as_logical_slice::<u16>().unwrap();
        assert_eq!(
            a, b,
            "actual BF16 output bits differ rows={rows} invalid={invalid}"
        );
        if !invalid {
            assert!(a.iter().all(|&v| half::bf16::from_bits(v).is_finite()));
            assert!(a.iter().any(|&v| v != 0 && v != SENTINEL));
        }
        assert_eq!(
            old.invalid_global.as_logical_slice::<u32>().unwrap(),
            new.invalid_global.as_logical_slice::<u32>().unwrap()
        );
        assert_eq!(
            old.invalid_heads.as_logical_slice::<u32>().unwrap(),
            new.invalid_heads.as_logical_slice::<u32>().unwrap()
        );
        if invalid {
            assert!(new
                .invalid_global
                .as_logical_slice::<u32>()
                .unwrap()
                .iter()
                .any(|&v| v != 0));
        } else {
            assert!(new
                .invalid_global
                .as_logical_slice::<u32>()
                .unwrap()
                .iter()
                .all(|&v| v == 0));
        }
        for output in [&old_output, &new_output] {
            let all = output.as_logical_slice::<u16>().unwrap();
            assert!(
                all[..H * D]
                    .iter()
                    .chain(&all[(rows + 1) * H * D..])
                    .all(|&v| v == SENTINEL),
                "output view guards changed"
            );
        }
        if rows == 257 && !invalid {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin_recorded().unwrap();
            assert!(encode_deepseek_sparse_flash_prefill(
                &mut session,
                registry,
                &device,
                &q,
                Some(&prefix),
                &kv,
                &compressed,
                &sinks,
                &ix,
                &new_view,
                &new,
                rows,
                PREFIX,
                PREFIX + rows + 1,
                PREFIX + rows + COMP,
                K,
                H,
                D,
                1.0
            )
            .is_err());
            let short = ix.slice_view(0, rows * K - 1);
            assert!(encode_deepseek_sparse_flash_prefill(
                &mut session,
                registry,
                &device,
                &q,
                Some(&prefix),
                &kv,
                &compressed,
                &sinks,
                &short,
                &new_view,
                &new,
                rows,
                PREFIX,
                PREFIX + rows,
                PREFIX + rows + COMP,
                K,
                H,
                D,
                1.0
            )
            .is_err());
            // Both errors precede the first barrier/copy; never submit this session.
            drop(session);
        }
        println!("DS_SPARSE_GATHER_TILE_CASE_COMPLETE rows={rows} invalid={invalid} exact_bits=true guards=true");
    }
    println!("DS_SPARSE_GATHER_TILE_COMPLETE cases=7 exact_bits=true requested_payload_lt_128mib=true model_loaded=false");
}
