use super::*;
use mlx_native::MlxDevice;
use std::cell::Cell;

fn config() -> Deepseek4Config {
    Deepseek4Config {
        num_hidden_layers: 3,
        hidden_size: 256,
        hidden_size_out: 1024,
        max_position_embeddings: 262144,
        vocab_size: 32,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 32,
        rope_head_dim: 8,
        rope_theta: 10000.0,
        rope_factor: 1.0,
        original_context_length: 128,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        q_lora_rank: 32,
        o_lora_rank: 32,
        output_groups: 2,
        sliding_window: 128,
        compress_ratios: vec![0, 4, 128],
        compress_rope_theta: 160000.0,
        index_num_heads: 2,
        index_head_dim: 16,
        index_top_k: 4,
        rms_norm_eps: 1e-6,
        num_experts: 4,
        num_experts_per_tok: 2,
        num_shared_experts: 1,
        expert_intermediate_size: 32,
        route_scale: 1.5,
        normalize_topk: true,
        swiglu_clamp_experts: vec![10.0; 3],
        swiglu_clamp_shared: vec![10.0; 3],
        hyper_connection_count: 4,
        hyper_connection_sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: 1,
    }
}
fn pristine(capacity: usize) -> BootstrapState {
    BootstrapState {
        capacity,
        position: 0,
        poisoned: false,
        grafted: false,
        published: false,
        empty: [true; 8],
    }
}
fn reserve<T>(items: &mut Vec<T>, count: usize) -> Result<()> {
    items.try_reserve_exact(count).context("test slot metadata")
}
#[test]
fn ds_bootstrap_plan_uses_actual_selected_descriptors() {
    let mut cfg = config();
    cfg.num_hidden_layers = 43;
    cfg.head_dim = 512;
    cfg.index_head_dim = 128;
    cfg.compress_ratios = (0..43)
        .map(|i| {
            if i < 2 {
                0
            } else if i % 2 == 0 {
                4
            } else {
                128
            }
        })
        .collect();
    let layout = Layout::new(&cfg, 262144, 4, &pristine(131072)).unwrap();
    assert_eq!(layout.initial.resident_bytes, 919617536);
    assert_eq!(layout.small.as_ref().unwrap().resident_bytes, 18722816);
    let full = Deepseek4CachePlan::for_context(&cfg, 262144).unwrap();
    assert_eq!(full.resident_bytes, 1821392896);
    assert_eq!(
        layout.initial.resident_bytes - layout.small.unwrap().resident_bytes,
        900894720
    );
}
#[test]
fn ds_bootstrap_one_and_four_slots_stage_then_transfer_without_extra_agent() {
    for count in [1, 4] {
        let layout = Layout::new(&config(), 262144, count, &pristine(131072)).unwrap();
        let calls = Cell::new(0);
        let mut slots = stage(
            &layout,
            reserve,
            |i, p, eager| {
                calls.set(calls.get() + 1);
                assert_eq!(
                    (p.context_length, eager),
                    if i == 0 { (128, true) } else { (131072, false) }
                );
                Ok(i + 1)
            },
            |_, _, _| Ok(()),
        )
        .unwrap();
        let mut boot = 999;
        transfer(&mut boot, &mut slots, |id| id);
        assert_eq!(boot, 1);
        assert_eq!(slots[0], 999);
        assert_eq!(calls.get(), count as usize);
        assert_eq!(
            slots
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            count as usize
        );
    }
}
#[test]
fn ds_bootstrap_every_nonpristine_field_and_existing_publication_falls_back() {
    let cfg = config();
    for fault in 0..13 {
        let mut state = pristine(131072);
        match fault {
            0..=7 => state.empty[fault] = false,
            8 => state.position = 1,
            9 => state.poisoned = true,
            10 => state.grafted = true,
            11 => state.published = true,
            _ => state.capacity = 128,
        }
        let layout = Layout::new(&cfg, 262144, 4, &state).unwrap();
        assert!(!layout.reuse);
        let slots = stage(
            &layout,
            reserve,
            |_, p, eager| {
                assert_eq!((p.context_length, eager), (131072, false));
                Ok(())
            },
            |_, _, _| Ok(()),
        )
        .unwrap();
        assert_eq!(slots.len(), 4);
    }
}
#[test]
fn ds_bootstrap_failures_never_reach_transfer_or_publication() {
    let layout = Layout::new(&config(), 262144, 4, &pristine(131072)).unwrap();
    let calls = Cell::new(0);
    assert!(stage::<usize>(
        &layout,
        |_, _| anyhow::bail!("reserve fault"),
        |_, _, _| {
            calls.set(1);
            Ok(0)
        },
        |_, _, _| Ok(())
    )
    .is_err());
    assert_eq!(calls.get(), 0);
    for validation in [false, true] {
        for fault in 0..4 {
            let mut boot = 999;
            let mut published = None;
            let result = stage(
                &layout,
                reserve,
                |i, _, _| {
                    ensure!(validation || i != fault, "allocation fault");
                    Ok(i)
                },
                |i, _, _| {
                    ensure!(!validation || i != fault, "validation fault");
                    Ok(())
                },
            );
            if let Ok(mut staged) = result {
                transfer(&mut boot, &mut staged, |id| id);
                published = Some(staged);
            }
            assert_eq!(boot, 999);
            assert!(published.is_none());
        }
    }
}
#[test]
fn ds_bootstrap_context_edges_and_repeated_fallback_preserve_capacities() {
    for context in [128, 1024, 131072, 262144] {
        let initial = context.min(131072);
        let layout = Layout::new(&config(), context, 1, &pristine(initial)).unwrap();
        assert_eq!(layout.initial.context_length, initial);
        assert_eq!(layout.small.unwrap().context_length, 128);
        let mut state = pristine(128);
        state.published = true;
        let repeated = Layout::new(&config(), context, 1, &state).unwrap();
        assert!(!repeated.reuse);
        assert_eq!(repeated.allocation(0).0.context_length, initial);
    }
    assert!(Layout::new(&config(), 262144, 0, &pristine(131072)).is_err());
}

#[test]
fn ds_bootstrap_anomalous_reservation_rejects_before_cache_allocation() {
    let layout = Layout::new(&config(), 262144, 4, &pristine(131072)).unwrap();
    let called = Cell::new(false);
    assert!(stage::<usize>(
        &layout,
        |_, _| Ok(()),
        |_, _, _| {
            called.set(true);
            Ok(0)
        },
        |_, _, _| Ok(())
    )
    .is_err());
    assert!(!called.get());
    assert!(Layout::new(&config(), 0, 4, &pristine(0)).is_err());
}

fn physical_roots(cache: &Deepseek4Cache) -> impl Iterator<Item = &mlx_native::MlxBuffer> {
    cache.layers().iter().flat_map(|l| {
        [
            Some(&l.attention_kv),
            l.indexer_kv.as_ref(),
            l.main_kv_state.as_ref(),
            l.main_score_state.as_ref(),
            l.indexer_kv_state.as_ref(),
            l.indexer_score_state.as_ref(),
        ]
        .into_iter()
        .flatten()
    })
}
fn pattern(index: usize, layer: usize, role: usize, seed: usize) -> u8 {
    ((index / 251 + index % 251 + layer * 17 + role * 29 + seed) % 256) as u8
}
fn fill(cache: &Deepseek4Cache, seed: usize) {
    for (l, layer) in cache.layers().iter().enumerate() {
        for (r, root) in [
            Some(&layer.attention_kv),
            layer.indexer_kv.as_ref(),
            layer.main_kv_state.as_ref(),
            layer.main_score_state.as_ref(),
            layer.indexer_kv_state.as_ref(),
            layer.indexer_score_state.as_ref(),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(root) = root {
                // A temporary wrapper clone, never a new payload allocation.
                let mut writable = root.clone();
                for (i, byte) in writable
                    .as_mut_slice::<u8>()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *byte = pattern(i, l, r, seed);
                }
            }
        }
    }
}
fn identity(cache: &Deepseek4Cache) -> usize {
    cache.layers()[0]
        .attention_kv
        .as_slice::<u8>()
        .unwrap()
        .as_ptr() as usize
}
fn check_valid(cache: &Deepseek4Cache, position: usize, seed: usize) {
    for (l, layer) in cache.layers().iter().enumerate() {
        let ratio = config().compress_ratios[l] as usize;
        let window_bytes = 32 * 2 * position.min(128);
        let compressed_bytes = if ratio == 0 {
            0
        } else {
            position / ratio * 32 * 2
        };
        let indexer_bytes = if ratio == 4 { position / 4 * 16 * 2 } else { 0 };
        for (root, length, offset, role) in [
            (Some(&layer.window_kv), window_bytes, 0, 0),
            (
                layer.compressed_kv.as_ref(),
                compressed_bytes,
                128 * 32 * 2,
                0,
            ),
            (layer.indexer_kv.as_ref(), indexer_bytes, 0, 1),
        ] {
            if let Some(root) = root {
                for (i, &byte) in root.as_slice::<u8>().unwrap()[..length].iter().enumerate() {
                    assert_eq!(byte, pattern(offset + i, l, role, seed));
                }
            }
        }
        for (role, root) in [
            (2, layer.main_kv_state.as_ref()),
            (3, layer.main_score_state.as_ref()),
            (4, layer.indexer_kv_state.as_ref()),
            (5, layer.indexer_score_state.as_ref()),
        ] {
            if let Some(root) = root {
                for (i, &byte) in root.as_slice::<u8>().unwrap().iter().enumerate() {
                    assert_eq!(byte, pattern(i, l, role, seed));
                }
            }
        }
    }
}
#[test]
#[ignore = "requires CPU-accessible Metal; no GPU commands or model"]
fn ds_bootstrap_actual_cpu_transfer_restore_growth() {
    let cfg = config();
    let initial = Deepseek4CachePlan::for_context(&cfg, 256).unwrap();
    let grown_plan = Deepseek4CachePlan::for_context(&cfg, 512).unwrap();
    assert_eq!(
        (initial.resident_bytes, grown_plan.resident_bytes),
        (69760, 76032)
    );
    let device = MlxDevice::new().expect("Metal required; no skip");
    for count in [1, 4] {
        let layout = Layout::new(&cfg, 256, count, &pristine(256)).unwrap();
        assert_eq!(layout.small.as_ref().unwrap().resident_bytes, 66624);
        let mut boot = Deepseek4Cache::allocate(&initial, device.clone()).unwrap();
        fill(&boot, 77);
        let original = identity(&boot);
        let mut slots = stage(
            &layout,
            reserve,
            |i, p, eager| {
                let cache = if eager {
                    Deepseek4Cache::allocate(p, device.clone())
                } else {
                    Deepseek4Cache::allocate_logical(p, device.clone())
                }?;
                fill(&cache, if i == 0 { 99 } else { 10 + i });
                Ok(cache)
            },
            |_, cache, p| validate_cache(cache, p),
        )
        .unwrap();
        let replacement = identity(&slots[0]);
        transfer(&mut boot, &mut slots, |cache| cache);
        assert_eq!(identity(&boot), replacement);
        assert_eq!(identity(&slots[0]), original);
        assert_eq!(boot.capacity(), 128);
        assert!(slots.iter().all(|cache| cache.capacity() == 256));
        let mut identities: Vec<_> = slots.iter().map(identity).collect();
        identities.push(identity(&boot));
        assert_eq!(
            identities
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            count as usize + 1
        );
        assert_eq!(physical_roots(&boot).count(), 10);
        assert!(slots
            .iter()
            .all(|cache| physical_roots(cache).count() == 10));
        // Existing request swap bracket, with the same production commit helper.
        transfer(&mut boot, &mut slots, |cache| cache);
        assert_eq!(identity(&boot), original);
        for position in 0..64 {
            boot.commit_step(position).unwrap();
        }
        let mut snapshot = boot.snapshot().unwrap();
        assert_eq!(snapshot.resident_bytes(), 63488);
        // Mutate only rollback-owned regions. Compressed/indexer rows are append-only.
        for layer in boot.layers() {
            for root in [
                Some(&layer.window_kv),
                layer.main_kv_state.as_ref(),
                layer.main_score_state.as_ref(),
                layer.indexer_kv_state.as_ref(),
                layer.indexer_score_state.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                root.clone().as_mut_slice::<u8>().unwrap().fill(0);
            }
        }
        boot.commit_step(64).unwrap();
        boot.restore(&snapshot).unwrap();
        assert_eq!(boot.position(), 64);
        check_valid(&boot, 64, 77);
        let mut grown = Deepseek4Cache::allocate_logical(&grown_plan, device.clone()).unwrap();
        grown.migrate_from(&boot, Some(&mut snapshot)).unwrap();
        assert_eq!((grown.position(), grown.capacity()), (64, 512));
        grown.restore(&snapshot).unwrap();
        check_valid(&grown, 64, 77);
        boot = grown;
        transfer(&mut boot, &mut slots, |cache| cache);
        assert_eq!((boot.capacity(), boot.position()), (128, 0));
        assert_eq!(identity(&boot), replacement);
        check_valid(&boot, 128, 99);
        assert_eq!((slots[0].capacity(), slots[0].position()), (512, 64));
        check_valid(&slots[0], 64, 77);
        for (i, cache) in slots.iter().enumerate().skip(1) {
            check_valid(cache, 128, 10 + i);
        }
        println!("DS_BOOTSTRAP_CPU_CASE slots={count} transfer=true rollback=true growth=true isolated=true gpu_submissions=0");
    }
    println!("DS_BOOTSTRAP_CPU_COMPLETE cases=2 planned_roots=108 historical_requested_bytes=761088 maximum_live_requested_bytes=485184 numeric_host_bound=16384 gpu_submissions=0 model_loaded=false runtime_admission=false");
}
