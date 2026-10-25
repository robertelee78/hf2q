//! Pure descriptor/traversal controls. No device or model construction.
use super::*;
use crate::serve::config::LayerType;
use std::cell::RefCell;

fn cfg() -> Gemma4Config {
    // Exact scalar mapping from frozen GGUF bf65b7e1...: block_count30,
    // head_count_kv[8,8,8,8,8,2] repeated; swa/global key_length256/512,
    // window1024/context262144. No tensor/tokenizer payload fixture.
    Gemma4Config {
        vocab_size: 262144,
        hidden_size: 2816,
        intermediate_size: 2112,
        moe_intermediate_size: 704,
        num_hidden_layers: 30,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_global_key_value_heads: 2,
        head_dim: 256,
        global_head_dim: 512,
        rms_norm_eps: 1e-6,
        rope_theta_sliding: 10000.,
        rope_theta_global: 1000000.,
        sliding_window: 1024,
        max_position_embeddings: 262144,
        final_logit_softcapping: Some(30.),
        attention_bias: false,
        attention_k_eq_v: true,
        tie_word_embeddings: true,
        num_experts: 128,
        top_k_experts: 8,
        layer_types: (0..30)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            })
            .collect(),
    }
}
fn modes() -> Modes {
    Modes {
        hybrid: true,
        ..Modes::plain(DType::F32)
    }
}
fn all_modes() -> Modes {
    Modes {
        dense: true,
        nibble: true,
        ..modes()
    }
}
fn slot(phase: Phase, hd: usize, ring: bool, modes: Modes) -> SlotPlan {
    SlotPlan::new(phase, 7, 2, hd, 3, ring, 4, modes, None).unwrap()
}
#[test]
fn gemma_scaffold_selected_header_cells_and_actual_descriptor_traversal() {
    let plan = ScaffoldPlan::new(&cfg(), 30, 4, modes()).unwrap();
    assert_eq!((plan.bytes, plan.roots), (28_153_713_600, 210));
    assert_eq!(
        plan.phases.iter().map(|p| p.phase).collect::<Vec<_>>(),
        vec![Phase::Hb, Phase::Hybrid]
    );
    assert_eq!(
        plan.phases[0].slots.iter().map(|s| s.bytes).sum::<usize>(),
        11_331_216_000
    );
    assert_eq!(
        plan.phases[1].slots.iter().map(|s| s.bytes).sum::<usize>(),
        16_822_497_600
    );
    let counts = RefCell::new(Vec::new());
    let publications = RefCell::new(Vec::new());
    plan.execute(
        |s| {
            s.allocate_with(|d| {
                counts.borrow_mut().push((
                    s.phase,
                    s.layer,
                    d.error.clone(),
                    d.dtype,
                    d.shape.clone(),
                    d.zeroed,
                    d.bytes,
                ));
                Ok(d.bytes)
            })
        },
        |phase, values| {
            publications.borrow_mut().push((phase, values.len()));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(counts.borrow().len(), 210);
    assert_eq!(
        *publications.borrow(),
        vec![(Phase::Hb, 30), (Phase::Hybrid, 30)]
    );
    for (phase_index, expected) in [
        (0, [17_039_360, 2_181_046_400]),
        (1, [25_296_896, 3_238_015_040]),
    ] {
        let slots = &plan.phases[phase_index].slots;
        assert_eq!(
            (slots[0].cap, slots[0].ring, slots[0].bytes),
            (1024, true, expected[0])
        );
        assert_eq!(
            (slots[5].cap, slots[5].ring, slots[5].bytes),
            (262145, false, expected[1])
        );
        assert!(slots[0].buffers.iter().all(|d| d.zeroed));
        assert!(slots[5].buffers.iter().all(|d| !d.zeroed));
    }
    let hb = &plan.phases[0].slots[5].buffers;
    assert_eq!(
        hb.iter().map(|d| d.dtype).collect::<Vec<_>>(),
        vec![DType::U8, DType::F32, DType::U8, DType::F32]
    );
    assert_eq!(hb[0].shape, vec![4, 2, 262145, 512]);
    assert_eq!(hb[1].shape, vec![4, 2, 262145, 2]);
    assert_eq!(
        hb.iter().map(|d| d.error.as_str()).collect::<Vec<_>>(),
        vec![
            "hb_kv L5 K packed",
            "hb_kv L5 K norms",
            "hb_kv L5 V packed",
            "hb_kv L5 V norms"
        ]
    );
    let hybrid = &plan.phases[1].slots[5].buffers;
    assert_eq!(
        hybrid.iter().map(|d| d.dtype).collect::<Vec<_>>(),
        vec![DType::F16, DType::U8, DType::F32]
    );
}
#[test]
fn gemma_scaffold_modes_coexist_and_skipped_fields_are_not_published() {
    let cfg = cfg();
    let mut old = [10, 20, 30, 40];
    let plan = ScaffoldPlan::new(&cfg, 1, 4, Modes::plain(DType::F32)).unwrap();
    plan.execute(
        |s| s.allocate_with(|d| Ok(d.bytes)),
        |p, v| {
            assert_eq!(p, Phase::Hb);
            old[0] = v.len();
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(old, [1, 20, 30, 40]);
    let plan = ScaffoldPlan::new(&cfg, 1, 4, all_modes()).unwrap();
    assert_eq!(
        plan.phases.iter().map(|p| p.phase).collect::<Vec<_>>(),
        vec![Phase::Hb, Phase::Hybrid, Phase::Dense, Phase::Nibble]
    );
    assert_eq!(plan.roots, 13);
}
#[test]
fn gemma_scaffold_enabled_empty_phases_publish_empty() {
    let plan = ScaffoldPlan::new(&cfg(), 0, 4, all_modes()).unwrap();
    let mut phases = Vec::new();
    plan.execute::<()>(
        |_| panic!("empty plan must not allocate"),
        |p, v| {
            assert!(v.is_empty());
            phases.push(p);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        phases,
        vec![Phase::Hb, Phase::Hybrid, Phase::Dense, Phase::Nibble]
    );
    assert_eq!((plan.bytes, plan.roots), (0, 0));
}
#[test]
fn gemma_scaffold_actual_publication_order_survives_descriptor_faults() {
    let plan = ScaffoldPlan::new(&cfg(), 2, 1, all_modes()).unwrap();
    // Every descriptor failure: partial phase never publishes; all prior phases do.
    for fail in 0..26 {
        let count = RefCell::new(0usize);
        let publications = RefCell::new(Vec::new());
        let result = plan.execute(
            |s| {
                s.allocate_with(|_| {
                    let mut n = count.borrow_mut();
                    let current = *n;
                    *n += 1;
                    anyhow::ensure!(current != fail, "injected actual descriptor failure");
                    Ok(current)
                })
            },
            |p, v| {
                publications.borrow_mut().push((p, v.len()));
                Ok(())
            },
        );
        assert!(result.is_err());
        let expected = if fail < 8 {
            vec![]
        } else if fail < 14 {
            vec![Phase::Hb]
        } else if fail < 18 {
            vec![Phase::Hb, Phase::Hybrid]
        } else {
            vec![Phase::Hb, Phase::Hybrid, Phase::Dense]
        };
        assert_eq!(
            publications
                .borrow()
                .iter()
                .map(|(p, _)| *p)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(*count.borrow(), fail + 1);
    }
}
#[test]
fn gemma_scaffold_invalid_later_phase_preflights_before_any_callback() {
    let mut cfg = cfg();
    cfg.head_dim = 255; // HB/hybrid valid, nibble invalid.
    let calls = RefCell::new(0);
    let result = ScaffoldPlan::new(&cfg, 1, 4, all_modes()).and_then(|p| {
        p.execute(
            |_| {
                *calls.borrow_mut() += 1;
                Ok(())
            },
            |_, _| {
                *calls.borrow_mut() += 1;
                Ok(())
            },
        )
    });
    assert!(result.is_err());
    assert_eq!(*calls.borrow(), 0);
}
#[test]
fn gemma_scaffold_hybrid_f16_dummy_and_xlen_order() {
    let plan = slot(
        Phase::Hybrid,
        512,
        false,
        Modes {
            full_f16_v: true,
            xlen: true,
            ..modes()
        },
    );
    assert_eq!(
        plan.buffers.iter().map(|d| d.dtype).collect::<Vec<_>>(),
        vec![DType::F16, DType::F16, DType::F32, DType::BF16, DType::BF16]
    );
    assert_eq!(
        (
            plan.buffers[2].shape.clone(),
            plan.buffers[2].bytes,
            plan.buffers[2].zeroed
        ),
        (vec![1], 4, true)
    );
    assert!(plan
        .buffers
        .iter()
        .enumerate()
        .all(|(i, d)| i == 2 || !d.zeroed));
    assert!(plan.buffers[3].error.ends_with("xlen K L7"));
    assert!(plan.buffers[4].error.ends_with("xlen V L7"));
    let off = slot(Phase::Hybrid, 256, true, modes());
    assert!(!off.xlen);
    assert_eq!(off.buffers.len(), 3);
}
#[test]
fn gemma_scaffold_norm_floor_and_nibble_shape_asymmetry() {
    assert_eq!(slot(Phase::Hb, 511, false, modes()).norms, 1);
    let a = slot(Phase::Nibble, 256, false, modes());
    assert_eq!(a.buffers[0].shape, vec![4, 2, 3, 128]);
    assert_eq!(a.buffers[1].shape, vec![4, 2, 3]);
    let b = SlotPlan::new(Phase::Nibble, 0, 2, 512, 3, false, 4, modes(), Some(3)).unwrap();
    assert_eq!(b.buffers[1].shape, vec![4, 2, 3, 3]);
    assert!(SlotPlan::new(Phase::Nibble, 0, 2, 255, 3, false, 4, modes(), None).is_err());
}
#[test]
fn gemma_scaffold_dense_preserves_dtype_and_per_slot_axes() {
    for dtype in [DType::F32, DType::F16, DType::U8] {
        let p = slot(Phase::Dense, 256, true, Modes::plain(dtype));
        assert_eq!(p.buffers.len(), 2);
        assert_eq!(p.buffers[0].dtype, dtype);
        assert_eq!(p.buffers[0].bytes, 4 * 2 * 3 * 256 * dtype.size_of());
    }
    for nseq in [1, 4] {
        let p = SlotPlan::new(Phase::Hb, 0, 8, 256, 1024, true, nseq, modes(), None).unwrap();
        assert_eq!(p.buffers[0].shape, vec![nseq as usize, 8, 1024, 256]);
    }
}
#[test]
fn gemma_scaffold_diagnostic_priority_and_overflow() {
    let error = SlotPlan::new(Phase::Hb, 0, 0, 0, 0, false, 0, modes(), None)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("n_seqs must be > 0"));
    assert!(ScaffoldPlan::new(&cfg(), 0, 0, modes())
        .err()
        .unwrap()
        .to_string()
        .contains("max_slots == 0"));
    assert!(SlotPlan::new(
        Phase::Hb,
        0,
        usize::MAX,
        512,
        262145,
        false,
        4,
        modes(),
        None
    )
    .is_err());
    assert!(BufferDescriptor::new(DType::F32, vec![usize::MAX], false, "overflow".into()).is_err());
    let mut huge = cfg();
    huge.max_position_embeddings = usize::MAX;
    assert_eq!(
        super::super::layer_type_to_alloc_params_per_slot(LayerType::Full, 1024, usize::MAX, 4).1,
        usize::MAX
    );
    assert!(ScaffoldPlan::new(&huge, 6, 4, modes()).is_err());
}
#[test]
fn gemma_scaffold_actual_schedule_not_block_count_and_descriptor_labels() {
    let mut cfg = cfg();
    cfg.num_hidden_layers = 999;
    cfg.layer_types = vec![LayerType::Full, LayerType::Sliding];
    let p = ScaffoldPlan::new(&cfg, 2, 4, modes()).unwrap();
    assert_eq!(p.phases[0].slots.len(), 2);
    assert_eq!(p.phases[0].slots[0].cap, 262145);
    assert_eq!(p.phases[0].slots[1].cap, 1024);
    for phase in &p.phases {
        for slot in &phase.slots {
            let seen = slot
                .allocate_with(|d| Ok((d.error.clone(), d.bytes, d.shape.clone())))
                .unwrap();
            assert_eq!(seen.len(), slot.buffers.len());
            assert_eq!(seen.iter().map(|v| v.1).sum::<usize>(), slot.bytes);
        }
    }
}

#[test]
fn gemma_scaffold_nibble_actual_traversal_preserves_error_contexts() {
    let plan = ScaffoldPlan::new(&cfg(), 1, 4, all_modes()).unwrap();
    for target in [
        "multi-seq MLX K packed L0",
        "multi-seq MLX K norms L0",
        "multi-seq MLX V packed L0",
        "multi-seq MLX V norms L0",
    ] {
        let publications = RefCell::new(Vec::new());
        let result = plan.execute(
            |slot| {
                slot.allocate_with(|descriptor| {
                    anyhow::ensure!(
                        descriptor.error != target,
                        "{}: synthetic native allocation failure",
                        descriptor.error
                    );
                    Ok(descriptor.error.clone())
                })
            },
            |phase, _| {
                publications.borrow_mut().push(phase);
                Ok(())
            },
        );
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains(target), "{error}");
        assert!(
            error
                .contains("ADR-040 iter-C2c-cont-cont: alloc_multi_seq_mlx_kv_for_layer L0 failed"),
            "{error}"
        );
        assert!(!error.contains("multi-seq MLX L0"));
        assert_eq!(
            *publications.borrow(),
            vec![Phase::Hb, Phase::Hybrid, Phase::Dense]
        );
    }
}
