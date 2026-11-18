// Parent dispatches ONE family/case/mode per process. This test never mutates the environment.
// Required: HF2Q_DFLASH2_NATIVE_AF4_FIXTURE and existing HF2Q_QWEN{36,38}_{GGUF,
// DFLASH2_CONFIG,DFLASH2_WEIGHTS}, HF2Q_DFLASH2_DRAFT_AF4_G64=1. Production staged
// selection stays external. HF2Q_DFLASH2_NATIVE_AF4_ACTUAL_OUT optionally names a NEW
// output file; otherwise a unique file is retained in the OS temporary directory.
type Af4Actual = std::collections::BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;

fn af4_record(actual: &mut Af4Actual, name: &str, shape: &[usize], values: &[f32]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == values.len(), "actual {name}: shape differs");
    actual.insert(name.to_string(), (Dtype::F32, shape.to_vec(), values.iter().flat_map(|x| x.to_le_bytes()).collect()));
    Ok(())
}

fn af4_record_ids(actual: &mut Af4Actual, name: &str, shape: &[usize], values: &[u32]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == values.len(), "actual {name}: ID shape differs");
    actual.insert(name.to_string(), (Dtype::I64, shape.to_vec(), values.iter().flat_map(|&x| i64::from(x).to_le_bytes()).collect()));
    Ok(())
}

fn af4_save_actual(actual: &Af4Actual, manifest_path: &Path, manifest: &serde_json::Value) -> Result<PathBuf> {
    let views = actual.iter().map(|(name, (dtype, shape, data))| {
        Ok((name.as_str(), TensorView::new(*dtype, shape.clone(), data)?))
    }).collect::<Result<Vec<_>>>()?;
    let path = std::env::var_os("HF2Q_DFLASH2_NATIVE_AF4_ACTUAL_OUT").map(PathBuf::from).unwrap_or_else(|| {
        std::env::temp_dir().join(format!("hf2q-native-af4-{}-{}-{}-{}.safetensors",
            manifest["family"].as_str().unwrap_or("unknown"), manifest["case"].as_str().unwrap_or("unknown"),
            std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))
    });
    ensure!(!path.exists(), "actual capture path already exists; never overwrite failed evidence: {}", path.display());
    let metadata = std::collections::HashMap::from([
        ("schema".to_string(), "hf2q.dflash2-native-af4-actual.v1".to_string()),
        ("reference_manifest_sha256".to_string(), oracle_file_sha256(manifest_path)?),
        ("semantic_state_route".to_string(), "original F32 captured-hidden projection/rebuild; uncompressed, not peer packed storage".to_string()),
        ("staged".to_string(), std::env::var("HF2Q_DFLASH2_DRAFT_STAGED").unwrap_or_default()),
    ]);
    safetensors::tensor::serialize_to_file(views, Some(metadata), &path)?;
    eprintln!("NATIVE_AF4_ACTUAL path={} sha256={}", path.display(), oracle_file_sha256(&path)?);
    Ok(path)
}

fn af4_compare_numeric(actual: &[f32], fixture: &SafeTensors<'_>, name: &str, dtype: Dtype,
    shape: &[usize], limit: f32, relative: bool, floor: bool) -> Result<()> {
    let expected = float_values(af4_tensor(fixture, name, dtype, shape)?);
    ensure!(actual.len() == expected.len() && actual.iter().all(|x| x.is_finite()), "{name}: invalid actual numeric shape/data");
    let delta = max_abs(actual, &expected);
    let scale = if relative { expected.iter().fold(if floor { 1.0_f32 } else { 0.0_f32 }, |m, x| m.max(x.abs())) } else { 1.0 };
    let error = delta / scale;
    eprintln!("NATIVE_AF4_NUMERIC {name} max_abs={delta} reference_scale={scale} error={error} limit={limit}");
    ensure!(error.is_finite() && error <= limit, "{name}: error={error} exceeds {limit}");
    Ok(())
}

fn af4_actual_values(actual: &Af4Actual, name: &str) -> Result<Vec<f32>> {
    let (_, shape, bytes) = actual.get(name).with_context(|| format!("actual snapshot missing {name}"))?;
    Ok(float_values(TensorView::new(Dtype::F32, shape.clone(), bytes)?))
}

fn af4_capture_graph(device: &MlxDevice, registry: &mut KernelRegistry, runtime: &mut DFlash2Runtime,
    adapter: &QwenDFlash2Adapter<'_>, fixture: &SafeTensors<'_>, anchors: &[u32], actual: &mut Af4Actual) -> Result<()> {
    let config = runtime.config.clone();
    let lanes = runtime.workspace.max_lanes;
    let positions = runtime.lane_lengths().iter().map(|&x| u32::try_from(x)).collect::<Result<Vec<_>, _>>()?;
    prepare_decode_positions(&mut runtime.workspace, &config, &positions)?;
    let noise = float_values(af4_tensor(fixture, "input.noise_embedding", Dtype::F32, &[lanes, 8, config.hidden_size])?);
    // Match the native fixture's one F32 -> BF16 input interface while filling
    // the existing F32 workspace; keep the original semantic input below.
    for (destination, &value) in runtime.workspace.hidden[0].as_mut_slice::<f32>()?.iter_mut().zip(&noise) {
        *destination = bf16::from_f32(value).to_f32();
    }
    af4_record(actual, "input.noise_embedding", &[lanes, 8, config.hidden_size], &noise)?;
    for layer in 0..config.num_hidden_layers {
        let snapshots = oracle_layer_snapshots(device, &config, lanes)?;
        let mut encoder = device.command_encoder()?;
        super::super::forward::dispatch_layer_inner(&mut encoder, registry, device, &config,
            &runtime.tensors.layers[layer], &runtime.workspace, layer, Some(&snapshots))?;
        encoder.commit_and_wait()?;
        if let Some(staged) = runtime.workspace.staged.as_mut() { staged.command_committed(); }
        // The layer does not overwrite its incoming ping-pong buffer.
        let incoming = runtime.workspace.hidden[layer & 1].as_slice::<f32>()?.to_vec();
        for (branch, base) in [("attention", 0), ("mlp", 3)] {
            for (stage, slot, width) in [("prepare", 0, config.hidden_size), ("dynamic.full", 1, config.convolution_groups() * 4),
                ("finish.fused", 2, config.hidden_size)] {
                af4_record(actual, &format!("output.layer.{layer}.{branch}.{stage}"), &[lanes, 8, width], snapshots[base + slot].as_slice::<f32>()?)?;
            }
            af4_record(actual, &format!("output.layer.{layer}.{branch}.finish.incoming"), &[lanes, 8, config.hidden_size],
                if base == 0 { &incoming } else { snapshots[2].as_slice::<f32>()? })?;
        }
        af4_record(actual, &format!("output.layer.{layer}.output"), &[lanes, 8, config.hidden_size],
            runtime.workspace.hidden[(layer + 1) & 1].as_slice::<f32>()?)?;
    }
    // Finish this same forward through the production final norm, actual target
    // head and selector. No replay, regenerated embedding, or alternate graph.
    runtime.workspace.uniforms.as_mut_slice::<f32>()?.fill(0.5);
    let prepared = super::super::staged::reads_prepared_head(&runtime.workspace, &config);
    let head = super::super::adapters::target_vocabulary_head(adapter, registry, device, prepared)?;
    let mut encoder = device.command_encoder()?;
    super::super::forward::dispatch_dflash2_finish(&mut encoder, registry, device, &config, &runtime.tensors, head, &mut runtime.workspace)?;
    super::super::selector::dispatch_dflash2_selection(&mut encoder, registry, device, &config, &runtime.tensors,
        &runtime.workspace, anchors, &vec![0.0; lanes], 0, &vec![8; lanes], &(0..lanes as u32).collect::<Vec<_>>(), None)?;
    encoder.commit_and_wait()?;
    if let Some(staged) = runtime.workspace.staged.as_mut() { staged.command_committed(); }
    let w = &runtime.workspace;
    af4_record(actual, "output.final_hidden", &[lanes, 8, config.hidden_size], w.final_hidden.as_slice::<f32>()?)?;
    let logits = w.logits.as_slice::<f32>()?;
    let selector = w.selector_hidden_bf16.as_slice::<u16>()?;
    let mut proposal_logits = Vec::with_capacity(lanes * 7 * config.vocab_size);
    let mut proposal_selector = Vec::with_capacity(lanes * 7 * config.selector_rank);
    for lane in 0..lanes {
        proposal_logits.extend_from_slice(&logits[(lane * 8 + 1) * config.vocab_size..(lane * 8 + 8) * config.vocab_size]);
        proposal_selector.extend(selector[(lane * 8 + 1) * config.selector_rank..(lane * 8 + 8) * config.selector_rank]
            .iter().map(|&x| bf16::from_bits(x).to_f32()));
    }
    af4_record(actual, "output.logits", &[lanes, 7, config.vocab_size], &proposal_logits)?;
    af4_record(actual, "output.selector_hidden", &[lanes, 7, config.selector_rank], &proposal_selector)?;
    af4_record_ids(actual, "output.candidates", &[lanes, 7, 16], w.candidates.as_slice::<u32>()?)?;
    af4_record_ids(actual, "output.path", &[lanes, 7], runtime.proposals()?)?;
    for (name, shape, buffer) in [("output.unary", vec![lanes, 7, 16], &w.unary),
        ("output.edges", vec![lanes, 7, 16, 16], &w.edges), ("output.confidence", vec![lanes, 7], &w.confidence)] {
        af4_record(actual, name, &shape, buffer.as_slice::<f32>()?)?;
    }
    for layer in 0..config.num_hidden_layers {
        for (kind, ring) in [("keys", &w.ring_keys[layer]), ("values", &w.ring_values[layer])] {
            let name = format!("state.layer.{layer}.{kind}");
            actual.insert(name, (Dtype::BF16, vec![lanes, config.num_key_value_heads, 2048, 128],
                ring.as_slice::<u16>()?.iter().flat_map(|x| x.to_le_bytes()).collect()));
        }
    }
    af4_record_ids(actual, "state.absolute_positions", &[lanes], &positions)?;
    Ok(())
}

fn af4_compare_outputs(device: &MlxDevice, registry: &mut KernelRegistry, runtime: &mut DFlash2Runtime,
    fixture: &SafeTensors<'_>, manifest: &serde_json::Value, actual: &Af4Actual, anchors: &[u32]) -> Result<()> {
    let c = runtime.config.clone();
    let lanes = anchors.len();
    let noise = af4_actual_values(actual, "input.noise_embedding")?;
    af4_compare_numeric(&noise, fixture, "native.input.noise_embedding", Dtype::BF16, &[lanes, 8, c.hidden_size], 0.03, false, false)?;
    for layer in 0..c.num_hidden_layers {
        for branch in ["attention", "mlp"] {
            for (stage, width) in [("prepare", c.hidden_size), ("dynamic.full", c.convolution_groups() * 4),
                ("finish.fused", c.hidden_size), ("finish.incoming", c.hidden_size)] {
                let name = format!("output.layer.{layer}.{branch}.{stage}");
                af4_compare_numeric(&af4_actual_values(actual, &name)?, fixture, &name, Dtype::BF16, &[lanes, 8, width], 0.05, true, true)?;
            }
        }
        let name = format!("output.layer.{layer}.output");
        af4_compare_numeric(&af4_actual_values(actual, &name)?, fixture, &name, Dtype::BF16, &[lanes, 8, c.hidden_size], 0.05, true, false)?;
    }
    for (name, dtype, shape, limit) in [("output.final_hidden", Dtype::BF16, vec![lanes, 8, c.hidden_size], 2.5),
        ("output.logits", Dtype::F32, vec![lanes, 7, c.vocab_size], 2.0),
        ("output.selector_hidden", Dtype::BF16, vec![lanes, 7, c.selector_rank], 1.1)] {
        af4_compare_numeric(&af4_actual_values(actual, name)?, fixture, name, dtype, &shape, limit, false, false)?;
    }
    let candidates = af4_tensor(fixture, "output.candidates", Dtype::I64, &[lanes, 7, 16])?;
    let expected_ids = i64_values(candidates).into_iter().map(u32::try_from).collect::<Result<Vec<_>, _>>()?;
    let ids = runtime.workspace.candidates.as_slice::<u32>()?;
    for (row, (a, e)) in ids.chunks_exact(16).zip(expected_ids.chunks_exact(16)).enumerate() {
        ensure!(a.iter().all(|&id| (id as usize) < c.vocab_size) && e.iter().all(|&id| (id as usize) < c.vocab_size), "candidate exceeds vocabulary");
        ensure!(a.iter().filter(|id| e.contains(id)).count() >= 10, "candidate overlap below 10/16 at row {row}");
    }
    let expected_path = i64_values(af4_tensor(fixture, "output.path", Dtype::I64, &[lanes, 7])?).into_iter().map(u32::try_from).collect::<Result<Vec<_>, _>>()?;
    for (lane, (a, e)) in runtime.proposals()?.chunks_exact(7).zip(expected_path.chunks_exact(7)).enumerate() {
        ensure!(a.iter().zip(e).filter(|(a,e)| a == e).count() >= 4, "path agreement below 4/7 lane={lane}");
    }
    af4_tensor(fixture, "output.unary", Dtype::F32, &[lanes, 7, 16])?;
    af4_tensor(fixture, "output.edges", Dtype::F32, &[lanes, 7, 16, 16])?;
    af4_tensor(fixture, "output.confidence", Dtype::F32, &[lanes, 7])?;
    af4_check_own_topk(runtime, actual)?;
    af4_compare_state(fixture, runtime)?;
    // Reuse the original token-aligned edge propagation/roundoff formulas and
    // conditional preceding-selected-token confidence on BOTH routes' own score tables.
    compare_selector_oracles(device, registry, runtime, fixture, anchors, manifest["case"].as_str().context("case missing")?)?;
    Ok(())
}

#[test]
fn native_af4_reference_fixture_conforms() {
    let Some(path) = std::env::var_os("HF2Q_DFLASH2_NATIVE_AF4_FIXTURE") else {
        eprintln!("SKIP native_af4_reference_fixture_conforms: HF2Q_DFLASH2_NATIVE_AF4_FIXTURE is unconfigured (not qualification)");
        return;
    };
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let path = PathBuf::from(path).canonicalize().expect("configured native-AF4 manifest must exist");
    let manifest = af4_json(&path).expect("read configured native-AF4 export manifest");
    let bytes = af4_verify_bundle(&manifest, path.parent().expect("manifest parent")).expect("verify fail-closed native-AF4 source/provenance/coverage");
    let family = manifest["family"].as_str().expect("family");
    let (family_env, descriptor, target_sha) = match family {
        "qwen36" => ("QWEN36", &QWEN36_DFLASH2, "815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220"),
        "qwen38" => ("QWEN38", &QWEN38_DFLASH2, "cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7"),
        _ => panic!("unsupported configured native-AF4 family"),
    };
    assert_eq!(
        manifest["bindings"]["peer_commit"], descriptor.reference_revision,
        "checkpoint descriptor must name the verified peer revision",
    );
    let env_path = |suffix: &str| {
        let name = format!("HF2Q_{family_env}_{suffix}");
        PathBuf::from(std::env::var_os(&name).unwrap_or_else(|| panic!("configured native-AF4 fixture requires {name}")))
    };
    let gguf_path = env_path("GGUF");
    let config_path = env_path("DFLASH2_CONFIG");
    let weights_path = env_path("DFLASH2_WEIGHTS");
    assert_eq!(manifest["bindings"]["target_gguf_sha256"], target_sha, "pinned target differs");
    for (file, key, expected) in [(&gguf_path, "target_gguf_sha256", target_sha),
        (&config_path, "checkpoint_config_sha256", descriptor.config_sha256),
        (&weights_path, "checkpoint_weights_sha256", descriptor.weights_sha256)] {
        assert_eq!(manifest["bindings"][key], expected, "configured {key} identity differs");
        assert_eq!(oracle_file_sha256(file).expect("hash configured model file"), expected, "configured {key} SHA differs");
    }
    let fixture = SafeTensors::deserialize(&bytes).expect("parse native-AF4 tensor payload");
    let config = descriptor.parse_config(&std::fs::read(&config_path).expect("read checkpoint config")).expect("parse checkpoint-declared layer count");
    let (lanes, rows, start) = af4_geometry(&manifest, &config).expect("validate every original geometry dimension");
    let anchors = i64_values(af4_tensor(&fixture, "input.anchor", Dtype::I64, &[lanes]).expect("exact anchors"))
        .into_iter().map(u32::try_from).collect::<Result<Vec<_>, _>>().expect("lossless anchor conversion");
    let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open actual target GGUF");
    let mut progress = LoadProgress::new(false, 1, 0);
    let model = Qwen35Model::load_from_gguf(&gguf, &mut progress)
        .expect("load actual target")
        .finalize_canonical_experts()
        .expect("finalize actual target experts before conformance caches");
    model.ensure_gpu_cache_primed().expect("prime actual target GPU cache");
    model.with_gpu_cache_mut(|device, registry| {
        let adapter = QwenDFlash2Adapter::new(&model, &config)?;
        let mut runtime = DFlash2Runtime::load_verified(device, descriptor, &config_path, &weights_path, &adapter, lanes)?;
        af4_check_weights(&manifest, &runtime, &adapter, &model, &gguf)?;
        af4_initialize_context(&mut runtime, device, registry, &fixture, rows, start)?;
        af4_validate_native_state(&manifest, &fixture, &runtime)?;
        let before = af4_ring_bits(&runtime)?;
        let lengths = runtime.lane_lengths().to_vec();
        let mut actual = Af4Actual::new();
        let staged = std::env::var("HF2Q_DFLASH2_DRAFT_STAGED").ok().as_deref() == Some("1");
        ensure!(runtime.workspace.staged.is_some() == staged, "configured staged route was not selected");
        super::super::staged::take_dispatch_counts();
        let execution = af4_capture_graph(device, registry, &mut runtime, &adapter, &fixture, &anchors, &mut actual);
        let dispatches = super::super::staged::take_dispatch_counts();
        af4_record_ids(&mut actual, "execution.staged_dispatches", &[4], &dispatches[..4])?;
        af4_record_ids(&mut actual, "execution.qkv_decode_dispatches", &[1], &dispatches[4..5])?;
        af4_record_ids(&mut actual, "execution.bf16_output_register_dispatches", &[1], &dispatches[5..])?;
        let retained = af4_save_actual(&actual, &path, &manifest)?;
        execution.with_context(|| format!("actual graph execution failed; partial capture {}", retained.display()))?;
        ensure!(af4_ring_bits(&runtime)? == before && runtime.lane_lengths() == lengths.as_slice(), "speculative draft changed persistent context bytes/cursors");
        let qkv_width = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim;
        let decode_rows = u32::try_from(lanes * config.block_size)?;
        let qkv_decode = decode_rows % 8 == 0
            && !mlx_native::ops::dflash2_af4g64_staged::dflash2_af4g64_decode_uses_register(
                decode_rows, u32::try_from(qkv_width)?,
            );
        let layers = u32::try_from(config.num_hidden_layers)?;
        ensure!(dispatches[4] == if qkv_decode { layers } else { 0 },
            "BF16 QKV producer did not encode its declared decode profile: {dispatches:?}");
        // Six BF16-producing projection calls per layer: both dynamic
        // projections, QKV, output, gate/up and down. Count actual encodes across
        // the source-selected families, not an obsolete forced-MPP threshold.
        ensure!(dispatches[0] + dispatches[1] + dispatches[4] + dispatches[5] == 6 * layers,
            "BF16 source plan missed a real projection encode: {dispatches:?}");
        if staged {
            ensure!(dispatches[1] == layers && dispatches[2] == layers,
                "staged route did not encode its fused gate/up and ring math: {dispatches:?}");
            ensure!(dispatches[3] == u32::from(lanes * config.block_size == 32),
                "prepared-head route differs from its actual row geometry: {dispatches:?}");
        } else {
            ensure!(dispatches[..4] == [0; 4], "auxiliary register route encoded auxiliary staged math: {dispatches:?}");
        }
        eprintln!("NATIVE_AF4_DISPATCHES qkv_decode={} auxiliary_staged_projection={} staged_fused_gate_up={} ring_attention={} prepared_head={} bf16_output_register={} head_route={}",
            dispatches[4], dispatches[0], dispatches[1], dispatches[2], dispatches[3], dispatches[5],
            if dispatches[3] == 0 { "stock" } else { "prepared" });
        af4_crop_continuation(&mut runtime, device, registry, &fixture, rows, start)?;
        af4_compare_outputs(device, registry, &mut runtime, &fixture, &manifest, &actual, &anchors)
            .with_context(|| format!("native-AF4 conformance rejected; retain {}", retained.display()))?;
        eprintln!("NATIVE_AF4_CONFORMS family={family} case={} lanes={lanes} staged={} actual={}",
            manifest["case"], runtime.workspace.staged.is_some(), retained.display());
        Ok(())
    }).expect("real source-bound native-AF4 conformance (no retry or fallback)");
}

fn af4_check_own_topk(runtime: &DFlash2Runtime, actual: &Af4Actual) -> Result<()> {
    let logits = af4_actual_values(actual, "output.logits")?;
    let ids = runtime.workspace.candidates.as_slice::<u32>()?;
    let unary = runtime.workspace.unary.as_slice::<f32>()?;
    let vocabulary = runtime.config.vocab_size;
    for (row, scores) in logits.chunks_exact(vocabulary).enumerate() {
        let selected = &ids[row * 16..(row + 1) * 16];
        for (column, &token) in selected.iter().enumerate() {
            ensure!(!selected[..column].contains(&token) && (token as usize) < vocabulary,
                "production top-k duplicate/out-of-range token");
            ensure!(unary[row * 16 + column].to_bits() == scores[token as usize].to_bits(),
                "production unary is not its actual selected-token logit");
            if column > 0 {
                let preceding = selected[column - 1];
                ensure!(scores[preceding as usize] > scores[token as usize]
                    || (scores[preceding as usize] == scores[token as usize] && preceding < token),
                    "production top-k ordering differs from score/token order");
            }
        }
        let tail = selected[15];
        let outranking = scores.iter().enumerate().filter(|(id, score)| **score > scores[tail as usize]
            || (**score == scores[tail as usize] && *id < tail as usize)).count();
        ensure!(outranking == 15, "production candidates are not its actual top-16 logits");
    }
    Ok(())
}
