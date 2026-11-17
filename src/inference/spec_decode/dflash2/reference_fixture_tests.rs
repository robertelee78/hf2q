    #[test]
    #[ignore = "requires pinned Qwen3.6 GGUF/checkpoint, HF2Q_QWEN36_DFLASH2_REFERENCE_FIXTURES, manifest and Apple Metal"]
    fn qwen36_fixture_suite_matches_pinned_reference() {
        fixture_suite_matches_pinned_reference(&QWEN36_DFLASH2, "QWEN36", "qwen36");
    }

    #[test]
    #[ignore = "requires pinned Qwen3.8 GGUF/checkpoint, HF2Q_QWEN38_DFLASH2_REFERENCE_FIXTURES, manifest and Apple Metal"]
    fn qwen38_fixture_suite_matches_pinned_reference() {
        fixture_suite_matches_pinned_reference(&QWEN38_DFLASH2, "QWEN38", "qwen38");
    }


    fn fixture_suite_matches_pinned_reference(
        descriptor: &'static DFlash2CheckpointDescriptor,
        family_env: &str,
        family: &str,
    ) {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let path = |suffix: &str| {
            let name = format!("HF2Q_{family_env}_{suffix}");
            PathBuf::from(std::env::var_os(&name).unwrap_or_else(|| panic!("{name}")))
        };
        let gguf_path = path("GGUF");
        let config_path = path("DFLASH2_CONFIG");
        let weights_path = path("DFLASH2_WEIGHTS");
        let fixture_directory = path("DFLASH2_REFERENCE_FIXTURES");
        let target_sha256 = oracle_file_sha256(&gguf_path).expect("hash reference target GGUF");
        let manifest = oracle_manifest(&path("DFLASH2_REFERENCE_MANIFEST"))
            .expect("read source-bound complete reference manifest");

        let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open reference target GGUF");
        let mut progress = LoadProgress::new(false, 1, 0);
        let model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load reference target");
        model
            .ensure_gpu_cache_primed()
            .expect("prime reference target GPU cache");
        model
            .with_gpu_cache_mut(|device, registry| {
                let config_bytes = std::fs::read(&config_path)?;
                let config = descriptor.parse_config(&config_bytes)?;
                let adapter = QwenDFlash2Adapter::new(&model, &config)?;
                let target_hidden = model.cfg.hidden_size as usize;

                for case in ["short", "wrap", "single", "multi", "restore"] {
                    let case_file = if case == "single" { "single1" } else { case };
                    let file_name = format!("{family}-{case_file}-f32.safetensors");
                    let fixture_path = fixture_directory.join(&file_name);
                    let fixture_bytes = std::fs::read(&fixture_path)
                        .with_context(|| format!("read {case} reference fixture"))?;
                    bind_reference_fixture(
                        descriptor, family, case, &file_name, &fixture_bytes,
                        &target_sha256, &manifest,
                    )?;
                    let (_, metadata) = SafeTensors::read_metadata(&fixture_bytes)
                        .with_context(|| format!("read {case} reference metadata"))?;
                    let metadata = metadata
                        .metadata()
                        .as_ref()
                        .context("reference metadata is missing")?;
                    ensure!(
                        metadata.get("schema").map(String::as_str)
                            == Some("hf2q.dflash2-reference-fixture.v2"),
                        "{case} fixture schema differs"
                    );
                    ensure!(
                        metadata.get("reference_commit").map(String::as_str)
                            == Some("95c8aeca5e4b4c4f9c0c967c05ab89fa3ed24f4c"),
                        "{case} fixture reference commit differs"
                    );
                    ensure!(
                        metadata.get("compute_dtype").map(String::as_str) == Some("float32"),
                        "{case} fixture must match hf2q's float32 activations"
                    );
                    ensure!(
                        metadata.get("case").map(String::as_str) == Some(case),
                        "{case} fixture metadata differs"
                    );
                    ensure!(
                        metadata.get("attention_precision").map(String::as_str)
                            == Some("bfloat16"),
                        "{case} fixture must match hf2q's packed attention precision"
                    );
                    let context_start = metadata
                        .get("context_start")
                        .context("context_start metadata is missing")?
                        .parse::<usize>()?;
                    let context_rows = metadata
                        .get("context_rows")
                        .context("context_rows metadata is missing")?
                        .parse::<usize>()?;
                    let fixture = SafeTensors::deserialize(&fixture_bytes)
                        .with_context(|| format!("parse {case} fixture"))?;

                    let target_view = fixture.tensor("input.target_hidden")?;
                    let target_shape = target_view.shape();
                    ensure!(
                        target_shape.len() == 3
                            && target_shape[1] == context_rows
                            && target_shape[2] == config.fc_input_dim(),
                        "{case} target input shape {target_shape:?} differs"
                    );
                    let lanes = target_shape[0];
                    let target = float_values(target_view);
                    let mut runtime = DFlash2Runtime::load_verified(
                        device,
                        descriptor,
                        &config_path,
                        &weights_path,
                        &adapter,
                        lanes,
                    )?;

                    if case == "restore" {
                        let initial_view = fixture.tensor("input.initial_target_hidden")?;
                        let initial_shape = initial_view.shape();
                        ensure!(
                            initial_shape.len() == 3
                                && initial_shape[0] == lanes
                                && initial_shape[2] == config.fc_input_dim(),
                            "restore initial target shape {initial_shape:?} differs"
                        );
                        let initial_rows = initial_shape[1];
                        let initial = float_values(initial_view);
                        for lane in 0..lanes {
                            let capture = capture_lane(
                                &initial,
                                lane,
                                initial_rows,
                                target_hidden,
                                &config.target_layer_ids,
                            );
                            runtime.ingest_target_capture(device, registry, &capture, lane, 0)?;
                        }
                    }
                    for lane in 0..lanes {
                        let capture = capture_lane(
                            &target,
                            lane,
                            context_rows,
                            target_hidden,
                            &config.target_layer_ids,
                        );
                        runtime.ingest_target_capture(
                            device,
                            registry,
                            &capture,
                            lane,
                            context_start,
                        )?;
                    }
                    ensure!(
                        runtime
                            .lane_lengths()
                            .iter()
                            .all(|length| *length == context_start + context_rows),
                        "{case} context lengths {:?} differ",
                        runtime.lane_lengths()
                    );

                    let anchors = i64_values(fixture.tensor("input.anchor")?)
                        .into_iter()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    let positions = runtime
                        .lane_lengths()
                        .iter()
                        .map(|length| u32::try_from(*length))
                        .collect::<Result<Vec<_>, _>>()?;
                    prepare_decode_positions(&mut runtime.workspace, &config, &positions)?;
                    let mut encoder = device.command_encoder()?;
                    dispatch_dflash2_input(
                        &runtime.tensors,
                        &mut encoder,
                        registry,
                        device,
                        &config,
                        &adapter,
                        &anchors,
                        &mut runtime.workspace,
                    )?;
                    encoder.commit_and_wait()?;
                    let expected_noise = float_values(fixture.tensor("input.noise_embedding")?);
                    let noise_delta = max_abs(
                        runtime.workspace.hidden[0].as_slice::<f32>()?,
                        &expected_noise,
                    );
                    for layer in 0..config.num_hidden_layers {
                        let mut encoder = device.command_encoder()?;
                        let snapshots = oracle_layer_snapshots(device, &config, lanes)?;
                        super::super::forward::dispatch_layer_inner(
                            &mut encoder,
                            registry,
                            device,
                            &config,
                            &runtime.tensors.layers[layer],
                            &runtime.workspace,
                            layer,
                            Some(&snapshots),
                        )?;
                        encoder.commit_and_wait()?;
                        // Compare the real forward snapshots against the original bounds.
                        if layer == 0 {
                            ensure!(noise_delta <= 0.03,
                                "{case} input embedding max_abs={noise_delta} exceeds 0.03");
                        }
                        compare_convolution_snapshots(
                            &fixture, &runtime, &snapshots, case, layer,
                        )?;
                        let expected =
                            float_values(fixture.tensor(&format!("output.layer.{layer}.output"))?);
                        let delta = max_abs(
                            runtime.workspace.hidden[(layer + 1) & 1].as_slice::<f32>()?,
                            &expected,
                        );
                        let expected_abs_max = expected
                            .iter()
                            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
                        ensure!(
                            delta / expected_abs_max <= 0.05,
                            "{case} layer {layer} relative max_abs={} exceeds 0.05",
                            delta / expected_abs_max
                        );
                    }

                    runtime.run_draft(device,
                    registry,
                    &adapter,
                    &anchors,
                    &positions,
                    &vec![0.0; lanes],
                    0,
                    &vec![8; lanes],
                    &vec![0.5; lanes * config.proposal_tokens()], None)?;

                    let expected_hidden = float_values(fixture.tensor("output.final_hidden")?);
                    let hidden_delta = max_abs(
                        runtime.workspace.final_hidden.as_slice::<f32>()?,
                        &expected_hidden,
                    );
                    ensure!(
                        hidden_delta <= 2.5,
                        "{case} final hidden max_abs={hidden_delta} exceeds 2.5"
                    );

                    let vocabulary = config.vocab_size;
                    let all_logits = runtime.workspace.logits.as_slice::<f32>()?;
                    let mut actual_logits =
                        Vec::with_capacity(lanes * config.proposal_tokens() * vocabulary);
                    for lane in 0..lanes {
                        let start = (lane * config.block_size + 1) * vocabulary;
                        actual_logits.extend_from_slice(
                            &all_logits[start..start + config.proposal_tokens() * vocabulary],
                        );
                    }
                    let expected_logits = float_values(fixture.tensor("output.logits")?);
                    let logits_delta = max_abs(&actual_logits, &expected_logits);
                    ensure!(
                        logits_delta <= 2.0,
                        "{case} logits max_abs={logits_delta} exceeds 2.0"
                    );

                    let all_selector = runtime.workspace.selector_hidden_bf16.as_slice::<u16>()?;
                    let mut actual_selector =
                        Vec::with_capacity(lanes * config.proposal_tokens() * config.selector_rank);
                    for lane in 0..lanes {
                        let start = (lane * config.block_size + 1) * config.selector_rank;
                        actual_selector.extend(
                            all_selector
                                [start..start + config.proposal_tokens() * config.selector_rank]
                                .iter()
                                .map(|bits| bf16::from_bits(*bits).to_f32()),
                        );
                    }
                    let expected_selector = float_values(fixture.tensor("output.selector_hidden")?);
                    let selector_delta = max_abs(&actual_selector, &expected_selector);
                    ensure!(
                        selector_delta <= 1.1,
                        "{case} selector hidden max_abs={selector_delta} exceeds 1.1"
                    );

                    let expected_candidates = i64_values(fixture.tensor("output.candidates")?)
                        .into_iter()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    let candidate_overlaps = runtime
                        .workspace
                        .candidates
                        .as_slice::<u32>()?
                        .chunks_exact(config.selector_top_k)
                        .zip(expected_candidates.chunks_exact(config.selector_top_k))
                        .map(|(actual, expected)| {
                            actual.iter().filter(|token| expected.contains(token)).count()
                        })
                        .collect::<Vec<_>>();
                    ensure!(
                        candidate_overlaps.iter().all(|overlap| *overlap >= 10),
                        "{case} candidate overlaps {candidate_overlaps:?} fall below 10/16"
                    );
                    let expected_path = i64_values(fixture.tensor("output.path")?)
                        .into_iter()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    let path_agreements = runtime
                        .proposals()?
                        .chunks_exact(config.proposal_tokens())
                        .zip(expected_path.chunks_exact(config.proposal_tokens()))
                        .map(|(actual, expected)| {
                            actual.iter().zip(expected).filter(|(a, e)| a == e).count()
                        })
                        .collect::<Vec<_>>();
                    ensure!(
                        path_agreements.iter().all(|agreement| *agreement >= 4),
                        "{case} path agreements {path_agreements:?} fall below 4/7: actual={:?} expected={expected_path:?}",
                        runtime.proposals()?
                    );
                    compare_selector_oracles(
                        device, registry, &mut runtime, &fixture, &anchors, case,
                    )?;
                }
                Ok(())
            })
            .expect("compare source-bound DFlash2 reference fixture suite");
    }

