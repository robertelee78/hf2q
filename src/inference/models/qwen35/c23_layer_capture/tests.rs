use super::*;

fn cfg() -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Dense, hidden_size: 5120, num_hidden_layers: 64,
        num_attention_heads: 4, num_key_value_heads: 2, head_dim: 32,
        linear_num_key_heads: 2, linear_num_value_heads: 2,
        linear_key_head_dim: 32, linear_value_head_dim: 32, linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        layer_types: super::super::default_layer_types(64, 4),
        partial_rotary_factor: 0.5, rope_theta: 10000.0, rotary_dim: 16,
        mrope_section: [4,4,0,0], mrope_interleaved: true, rms_norm_eps: 1e-6,
        max_position_embeddings: 4096, vocab_size: 128, attn_output_gate: true,
        mtp_num_hidden_layers: 0, mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(64), moe: None,
    }
}
fn owner() -> Owner {
    Owner { slot: SlotId(2), generation: 7, lane: 2, turn: 3, ticket: "synthetic-owner".into() }
}
fn store(root: &std::path::Path, cfg: &Qwen35Config, first: usize, end: usize) -> Store {
    let plan = expected(cfg, 2, 3, first, end).unwrap();
    let bytes = plan.iter().map(|entry| entry.shape.iter().product::<usize>() as u64 * 4).sum();
    Store::new(root, Identity { family: Family::Qwen38,
        source_commit: "a".repeat(40), binary_sha256: "b".repeat(64),
        native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(), native_checksum: "c".repeat(64),
        model_sha256: "d".repeat(64), tokenizer_sha256: "e".repeat(64),
        template_sha256: "f".repeat(64), config_sha256: "1".repeat(64),
        token_vectors_sha256: "2".repeat(64) }, "synthetic collector contract".into(),
        plan.clone(), bytes, plan.len()).unwrap()
}
#[test]
fn c23_full_geometry_has_fixed_packed_block_coverage() {
    let cfg = cfg();
    let plan = expected(&cfg, 2, 3, 914, 1923).unwrap();
    assert_eq!(plan.len(), 129);
    assert_eq!(layout(&cfg, 914, 1923).unwrap().2, 2_665_697_280);
    assert_eq!(plan.iter().map(|entry| entry.at.clone()).collect::<BTreeSet<_>>().len(), 129);
    for (index, entry) in plan.iter().enumerate() {
        let (layer, stage) = boundary(index);
        assert_eq!(point(64, layer, stage).unwrap(), index);
        assert_eq!(entry.shape, [1009,5120]);
        assert_eq!(entry.at.logical_row_start, 914);
        assert_eq!(entry.at.logical_step, 3);
        assert_eq!(entry.at.layer, layer as u32);
    }
    assert!(point(64,64,Stage::LayerOutput).is_err());
    assert!(point(64,1,Stage::Input).is_err());
    assert!(point(64,0,Stage::DenseOutput).is_err());
    assert!(point(64,0,Stage::Attention).is_err()); // no nonexistent raw intermediate claim
    assert!(expected(&cfg,2,3,914,914).is_err());
    assert!(expected(&cfg,2,3,0,MAX_ROWS+1).is_err());
    let mut overflow = cfg.clone(); overflow.hidden_size = u32::MAX;
    assert!(expected(&overflow,2,3,914,1923).is_err());
}
#[test]
fn c23_boundary_guard_rejects_duplicate_missing_and_unknown_points() {
    let mut points = BTreeSet::new();
    visit(&mut points,64,0,Stage::Input).unwrap();
    assert!(visit(&mut points,64,0,Stage::Input).is_err());
    assert!(!complete_points(&points,64));
    for layer in 0..64 {
        visit(&mut points,64,layer,Stage::Residual).unwrap();
        visit(&mut points,64,layer,Stage::LayerOutput).unwrap();
    }
    assert!(complete_points(&points,64));
    points.remove(&128); points.insert(129);
    assert!(!complete_points(&points,64));
}
#[test]
fn c23_owner_join_rejects_generation_ticket_and_row_aliases() {
    let owner = owner();
    let witness = serde_json::json!([
        {"physical_row":0,"logical_position":999,"owner":{"slot":0}},
        {"physical_row":1,"logical_position":914,"owner":{
            "slot":2,"generation":7,"lane":2,"turn":3,"ticket":"synthetic-owner"}}
    ]);
    validate_cohort_owner(&owner,&[(1,914)],&witness).unwrap();
    for (field, value) in [("generation",serde_json::json!(8)),("ticket",serde_json::json!("reused")),
        ("slot",serde_json::json!(3)),("turn",serde_json::json!(4))] {
        let mut stale = witness.clone(); stale[1]["owner"][field] = value;
        assert!(validate_cohort_owner(&owner,&[(1,914)],&stale).is_err());
    }
    assert!(validate_cohort_owner(&owner,&[(1,915)],&witness).is_err());
    assert!(validate_cohort_owner(&owner,&[(0,914)],&witness).is_err());
    assert!(validate_cohort_owner(&owner,&[(1,914)],&serde_json::Value::Null).is_err());
}
#[test]
fn c23_missing_and_unknown_retirement_never_finish_successfully() {
    let temp = tempfile::tempdir().unwrap();
    let parent = temp.path().canonicalize().unwrap();
    install(owner(),store(&parent.join("missing"),&cfg(),914,1923),914,1923).unwrap();
    assert!(finish().is_err());
    install(owner(),store(&parent.join("unknown"),&cfg(),914,1923),914,1923).unwrap();
    ACTIVE.with(|cell| {
        cell.borrow_mut().as_mut().unwrap().pending = Some(Pending {
            id:1,rows:vec![(0,914)],points:BTreeSet::new(),holds:vec![] });
    });
    drop(Forward { id:1,drained:false,_thread:std::marker::PhantomData });
    assert!(ACTIVE.with(|cell| cell.borrow().as_ref().unwrap().failed));
    assert!(finish().is_err());
    assert!(!selected());
}

#[test]
#[ignore = "Small synthetic Metal endpoint control; root runs once after integration, no model load"]
fn c23_snapshot_preserves_fused_endpoint_and_strided_rows_before_scratch_reuse() {
    use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;
    let _lock = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("actual Metal required; no unavailable-device PASS");
    let mut registry = mlx_native::KernelRegistry::new();
    let width = 32;
    let rows = 4;
    let alloc = |n| device.alloc_buffer(n*4,DType::F32,vec![n]).unwrap();
    let mut residual = alloc(rows*width);
    let mut attention = alloc(rows*width);
    let mut weights = alloc(width);
    let mut overwrite = alloc(rows*width);
    let values = (0..rows*width).map(|index| (index as f32-30.0)/31.0).collect::<Vec<_>>();
    residual.as_mut_slice::<f32>().unwrap().copy_from_slice(&values);
    attention.as_mut_slice::<f32>().unwrap().fill(0.125);
    weights.as_mut_slice::<f32>().unwrap().fill(1.0);
    overwrite.as_mut_slice::<f32>().unwrap().fill(-2.0);
    let baseline_norm = alloc(rows*width);
    let baseline_sum = alloc(rows*width);
    let mut enc = device.command_encoder().unwrap();
    dispatch_fused_residual_norm_f32(&mut enc,&mut registry,device.metal_device(),
        &residual,&attention,&weights,&baseline_norm,Some(&baseline_sum),rows as u32,width as u32,1e-6).unwrap();
    enc.commit_and_wait_labeled("c23.synthetic.baseline.existing-terminal").unwrap();
    let baseline = baseline_norm.as_logical_slice::<f32>().unwrap().iter().map(|x|x.to_bits()).collect::<Vec<_>>();

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap().join("snapshots");
    let mut geometry = cfg(); geometry.hidden_size = width as u32; geometry.num_hidden_layers = 1;
    install(owner(),store(&root,&geometry,914,916),914,916).unwrap();
    let (_,_,bytes) = layout(&geometry,914,916).unwrap();
    // This synthetic command owns these real row mappings, rather than borrowing
    // a serving scheduler identity. Only begin's external scheduler join is not exercised.
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut(); let sink = active.as_mut().unwrap();
        sink.width = width; sink.layers = 1;
        sink.bank = Some(device.alloc_buffer(bytes,DType::F32,vec![3,2,width]).unwrap());
        sink.entered.extend([914,915]);
        sink.pending = Some(Pending { id:1,rows:vec![(1,914),(3,915)],points:BTreeSet::new(),holds:vec![] });
    });
    let scope = Forward { id:1,drained:false,_thread:std::marker::PhantomData };
    let normed = alloc(rows*width);
    let sum = alloc(rows*width);
    let mut enc = device.command_encoder().unwrap();
    snapshot(&mut enc,&residual,0,Stage::Input,||serde_json::json!({"synthetic":"incoming residual"})).unwrap();
    dispatch_fused_residual_norm_f32(&mut enc,&mut registry,device.metal_device(),
        &residual,&attention,&weights,&normed,Some(&sum),rows as u32,width as u32,1e-6).unwrap();
    snapshot(&mut enc,&sum,0,Stage::Residual,||serde_json::json!({"synthetic":"actual fused sum"})).unwrap();
    snapshot(&mut enc,&normed,0,Stage::LayerOutput,||serde_json::json!({"synthetic":"fused kernel endpoint, not model FFN"})).unwrap();
    enc.blit_copy_bytes(&overwrite,0,&residual,0,(rows*width*4) as u64).unwrap();
    enc.blit_copy_bytes(&overwrite,0,&sum,0,(rows*width*4) as u64).unwrap();
    enc.commit_and_wait_labeled("c23.synthetic.observed.existing-terminal").unwrap();
    scope.finish_after_checked_output_head().unwrap();
    let observed = normed.as_logical_slice::<f32>().unwrap().iter().map(|x|x.to_bits()).collect::<Vec<_>>();
    assert_eq!(baseline,observed,"snapshot blits changed real fused-kernel endpoint");
    assert_eq!(finish().unwrap()["records"],3);
    let raw = std::fs::read(root.join("tensor-000000.bin")).unwrap();
    let expected = [values[width..2*width].to_vec(),values[3*width..4*width].to_vec()].concat();
    assert_eq!(raw,bytemuck::cast_slice::<f32,u8>(&expected));
    let sum_raw = std::fs::read(root.join("tensor-000001.bin")).unwrap();
    let expected_sum = expected.iter().map(|x| x+0.125).collect::<Vec<_>>();
    assert_eq!(sum_raw,bytemuck::cast_slice::<f32,u8>(&expected_sum));
    assert!(residual.as_logical_slice::<f32>().unwrap().iter().all(|&x|x == -2.0));
    assert!(sum.as_logical_slice::<f32>().unwrap().iter().all(|&x|x == -2.0));
}
