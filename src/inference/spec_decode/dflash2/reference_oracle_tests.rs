// Included in runtime::tests to retain the existing real-checkpoint helpers.
// Predeclared, NOT measured: convolution prepare/dynamic/finish use 0.05 times
// max(1, reference max_abs), the existing layer-relative budget with an absolute
// floor for near-zero tensors. Finish excludes the fused residual, as upstream
// GroupedDynamicCausalConv.finish does. No output/path limits are changed.
const CONVOLUTION_RELATIVE_LIMIT: f32 = 0.05;
// Same score tables: only F32 softmax/reduction rounding, not drafter error.
const CONFIDENCE_SAME_TABLE_MAX_ABS: f32 = 1.0e-5;
const REFERENCE_MODEL_SHA256: &str =
    "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e";

type OracleManifest = serde_json::Value;

fn oracle_file_sha256(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 { break; }
        digest.update(&chunk[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn oracle_manifest(path: &Path) -> Result<OracleManifest> {
    let manifest: OracleManifest = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(manifest["schema"] == "hf2q.dflash2-oracle-manifest.v1", "oracle manifest schema differs");
    let source = std::env::var("HF2Q_DFLASH2_REFERENCE_SOURCE")
        .context("HF2Q_DFLASH2_REFERENCE_SOURCE must bind the capture source commit")?;
    ensure!(source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()), "invalid oracle source commit");
    ensure!(manifest["hf2q_source"] == source, "oracle source commit differs");
    let producer = hex::encode(Sha256::digest(include_bytes!("../../../../scripts/dflash2_reference_fixture.py")));
    ensure!(manifest["producer_sha256"] == producer, "oracle producer source differs from the compiled consumer");
    let fixtures = manifest["fixtures"].as_object().context("manifest fixtures missing")?;
    ensure!(fixtures.len() == 5, "manifest must bind all five cases");
    Ok(manifest)
}

fn oracle_tensor_sha256(bytes: &[u8]) -> Result<String> {
    ensure!(bytes.len() >= 8, "fixture header missing");
    let length = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    let base = length.checked_add(8).context("fixture header overflow")?;
    ensure!(base <= bytes.len(), "fixture header exceeds file");
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..base])?;
    let fields = header.as_object().context("fixture tensor directory missing")?;
    let mut names = fields.keys().filter(|name| name.as_str() != "__metadata__").collect::<Vec<_>>();
    names.sort();
    let mut digest = Sha256::new();
    for name in names {
        let tensor = &fields[name];
        let descriptor = serde_json::to_vec(&(name, &tensor["dtype"], &tensor["shape"]))?;
        let offsets = tensor["data_offsets"].as_array()
            .filter(|offsets| offsets.len() == 2).context("tensor extent missing")?;
        let offset = |index: usize| offsets[index].as_u64()
            .and_then(|value| usize::try_from(value).ok()).context("invalid tensor offset");
        let (start, end) = (offset(0)?, offset(1)?);
        ensure!(start <= end && base.checked_add(end).is_some_and(|end| end <= bytes.len()),
            "tensor extent exceeds fixture");
        digest.update(u64::try_from(descriptor.len())?.to_le_bytes());
        digest.update(descriptor);
        digest.update(u64::try_from(end - start)?.to_le_bytes());
        digest.update(&bytes[base + start..base + end]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn bind_reference_fixture(
    descriptor: &DFlash2CheckpointDescriptor,
    family: &str,
    case: &str,
    file_name: &str,
    bytes: &[u8],
    target_sha256: &str,
    manifest: &OracleManifest,
) -> Result<()> {
    let hash = hex::encode(Sha256::digest(bytes));
    ensure!(manifest["fixtures"][file_name] == hash,
        "{case}: fixture hash differs from the source-bound capture manifest");
    if family == "qwen36" {
        // Preserve every original tensor name/dtype/shape/numeric byte.
        // Original metadata named an unmeasured target; independently
        // recaptured fixtures must instead bind the actual target below.
        let frozen = match case {
            "short" => "d99481af8aedbdcb21999bcd00b462aa74347ac3862b120bea61f8f345c4ed83",
            "wrap" => "514fcab223e9fedbc48713f37909f48822e5916cc6a146756da5b3e457e917cb",
            "single" => "1fd36edede35b80fbf38c4f037fa8f27e1814ecda48181df6dc219e2d5adf3eb",
            "multi" => "eecd640040cf102743d40307b2c6dbf50cb053af7dc6cf7f0e8c0b07388848b9",
            "restore" => "4b0db8ad55e01e985a24f4e11f1d1cc5a87fcf29bced5926c4c5be69e5fa89c0",
            _ => bail!("unknown oracle case {case}"),
        };
        ensure!(oracle_tensor_sha256(bytes)? == frozen,
            "{case}: frozen Qwen3.6 tensor corpus changed");
    }
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let metadata = header.metadata().as_ref().context("fixture metadata missing")?;
    for (key, expected) in [
        ("checkpoint_config_sha256", descriptor.config_sha256),
        ("checkpoint_weights_sha256", descriptor.weights_sha256),
        ("reference_model_sha256", REFERENCE_MODEL_SHA256),
        ("target_gguf_sha256", target_sha256),
    ] {
        ensure!(metadata.get(key).map(String::as_str) == Some(expected), "{case}: {key} differs");
    }
    let (rows, start, offset, lanes) = match case {
        "short" => (3, 0, 0, 1),
        "wrap" => (2048, 0, 0, 1),
        "single" => (17, 0, 1, 1),
        "multi" => (17, 0, 0, 2),
        "restore" => (3, 12, 0, 1),
        _ => unreachable!(),
    };
    for (key, value) in [("context_rows", rows), ("context_start", start), ("lane_offset", offset)] {
        ensure!(metadata.get(key) == Some(&value.to_string()), "{case}: frozen {key} geometry differs");
    }
    for key in ["hf2q_source", "producer_sha256"] {
        ensure!(metadata.get(key).map(String::as_str) == manifest[key].as_str(), "{case}: {key} binding differs");
    }
    ensure!(metadata.get("descriptor_id").map(String::as_str) == Some(descriptor.id), "descriptor differs");
    ensure!(metadata.get("target_source_sha256").map(String::as_str) == descriptor.target_source_sha256,
        "target source receipt differs");
    ensure!(metadata.get("oracle_contract").map(String::as_str) == Some("hf2q.dflash2-aligned-oracle.v1"),
        "aligned oracle contract missing");
    let fixture = SafeTensors::deserialize(bytes)?;
    ensure!(fixture.tensor("input.target_hidden")?.shape()[0] == lanes, "{case}: lane geometry differs");
    let anchors = i64_values(fixture.tensor("input.anchor")?);
    ensure!(anchors == (0..lanes).map(|lane| 17 + (offset as i64 + lane as i64) * 101).collect::<Vec<_>>(),
        "{case}: frozen anchor IDs differ");
    eprintln!("D1 {family} {case} fixture_sha256={hash} target_sha256={target_sha256}");
    Ok(())
}

fn oracle_layer_snapshots(device: &MlxDevice, config: &DFlash2Config, lanes: usize) -> Result<[MlxBuffer; 6]> {
    let rows = lanes * config.block_size;
    let mut buffers = Vec::with_capacity(6);
    for index in 0..6 {
        let width = if index % 3 == 1 { config.convolution_groups() * 4 } else { config.hidden_size };
        buffers.push(device.alloc_buffer(rows * width * 4, mlx_native::DType::F32, vec![rows, width])?);
    }
    Ok(buffers.try_into().ok().expect("six convolution snapshots"))
}

fn compare_convolution_snapshots(
    fixture: &SafeTensors<'_>, runtime: &DFlash2Runtime, snapshots: &[MlxBuffer; 6], case: &str, layer: usize,
) -> Result<()> {
    let config = &runtime.config;
    let groups = config.convolution_groups();
    for (branch, base) in [("attention", 0), ("mlp", 3)] {
        for (stage, slot) in [("prepare", 0), ("dynamic", 1), ("finish", 2)] {
            let expected = float_values(fixture.tensor(&format!("output.layer.{layer}.{branch}.{stage}"))?);
            let snapshot = snapshots[base + slot].as_slice::<f32>()?;
            let actual = match stage {
                // Upstream prepare returns ONLY the finish half of its projected kernel.
                "dynamic" => snapshot.chunks_exact(groups * 4)
                    .flat_map(|row| row[groups * 2..].iter().copied()).collect::<Vec<_>>(),
                // The native finish fuses residual addition; compare the convolution itself.
                "finish" => {
                    let residual = if base == 0 {
                        runtime.workspace.hidden[layer & 1].as_slice::<f32>()?
                    } else { snapshots[2].as_slice::<f32>()? };
                    snapshot.iter().zip(residual).map(|(sum, residual)| sum - residual).collect()
                }
                _ => snapshot.to_vec(),
            };
            ensure!(actual.len() == expected.len() && actual.iter().chain(&expected).all(|x| x.is_finite()),
                "{case} layer {layer} {branch}.{stage}: invalid numeric shape/data");
            let scale = expected.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
            let relative = max_abs(&actual, &expected) / scale;
            ensure!(relative <= CONVOLUTION_RELATIVE_LIMIT,
                "{case} layer {layer} {branch}.{stage}: relative max_abs={relative} exceeds {CONVOLUTION_RELATIVE_LIMIT}");
            eprintln!("D1 {case} layer={layer} {branch}.{stage} relative_max_abs={relative}");
        }
    }
    Ok(())
}

// Pinned model.py CandidateSelector.select (lines 524-547): greedy conditional
// on the SELECTED TOKEN, not an incidental torch.topk(sorted=False) column.
// Confidence is max softmax(unary + edge | selected predecessor), temperature 1.
// The pinned Python selector returns no greedy confidence: this is an explicit
// derived oracle for the native verify-budget quantity, bound to frozen tables.
fn oracle_path_confidence(
    candidates: &[u32], unary: &[f32], edges: &[f32], path: &[u32], lanes: usize, positions: usize,
) -> Result<Vec<f32>> {
    const K: usize = 16;
    ensure!(candidates.len() == lanes * positions * K && unary.len() == candidates.len()
        && edges.len() == candidates.len() * K && path.len() == lanes * positions, "selector oracle shape differs");
    let mut confidence = Vec::with_capacity(path.len());
    for lane in 0..lanes {
        let mut predecessor = 0;
        for position in 0..positions {
            let row = lane * positions + position;
            let ids = &candidates[row * K..(row + 1) * K];
            ensure!(ids.iter().enumerate().all(|(i, id)| !ids[..i].contains(id)), "duplicate candidate IDs");
            let selected = ids.iter().position(|id| *id == path[row]).context("selected path token absent from table")?;
            let scores = (0..K).map(|column| (unary[row * K + column]
                + edges[(row * K + predecessor) * K + column]) as f64).collect::<Vec<_>>();
            ensure!(scores.iter().all(|s| s.is_finite()), "nonfinite selector score");
            let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // Both pinned and native selectors round the addition in F32.
            // Widen only those rounded scores for the independent softmax.
            let selected_score = scores[selected];
            ensure!(scores.iter().all(|s| *s <= selected_score),
                "path does not select a maximal score at lane={lane} position={position}");
            confidence.push((1.0 / scores.iter().map(|s| (s - best).exp()).sum::<f64>()) as f32);
            predecessor = selected;
        }
    }
    Ok(confidence)
}

fn dispatch_oracle_score_tables(
    device: &MlxDevice, registry: &mut KernelRegistry, runtime: &mut DFlash2Runtime,
    candidates: &[u32], unary: &[f32], edges: &[f32], anchors: &[u32],
) -> Result<()> {
    use mlx_native::ops::{dflash2_selector::{DFlash2SelectorParams, DFLASH2_PATH_KERNEL},
        encode_helpers::{as_bytes, encode_with_args, KernelArg}};
    use mlx_native::metal::MTLSize;
    runtime.workspace.candidates.as_mut_slice::<u32>()?.copy_from_slice(candidates);
    runtime.workspace.unary.as_mut_slice::<f32>()?.copy_from_slice(unary);
    runtime.workspace.edges.as_mut_slice::<f32>()?.copy_from_slice(edges);
    let mut anchor = [0; 4];
    anchor[..anchors.len()].copy_from_slice(anchors);
    let params = DFlash2SelectorParams { anchor, temperature: [1.0; 4], lanes: anchors.len() as u32,
        rows_per_lane: runtime.config.block_size as u32, vocabulary: runtime.config.vocab_size as u32, sampling_mask: 0 };
    let w = &runtime.workspace;
    let pipeline = registry.get_pipeline(DFLASH2_PATH_KERNEL, device.metal_device())?;
    let mut encoder = device.command_encoder()?;
    // This IS the native production path kernel, not a replacement selector.
    encode_with_args(&mut encoder, pipeline, &[
        (0, KernelArg::Buffer(&w.candidates)), (1, KernelArg::Buffer(&w.unary)),
        (2, KernelArg::Buffer(&w.edges)), (3, KernelArg::Buffer(&w.uniforms)),
        (4, KernelArg::Buffer(&w.proposed_tokens)), (5, KernelArg::Buffer(&w.proposal_probabilities)),
        (6, KernelArg::Buffer(&w.confidence)), (7, KernelArg::Bytes(as_bytes(&params))),
    ], MTLSize::new(anchors.len() as u64, 1, 1), MTLSize::new(1, 1, 1));
    encoder.commit_and_wait()?;
    Ok(())
}

fn oracle_aligned_edge(
    actual: f32, reference: Option<(f32, &[f32])>, native_hidden: &[u16],
    predecessor: &[u16], successor: &[u16],
) -> Result<Option<f64>> {
    let rank = native_hidden.len();
    ensure!(rank > 0 && predecessor.len() == rank && successor.len() == rank
        && reference.is_none_or(|(_, hidden)| hidden.len() == rank),
        "aligned edge input shape differs");
    let mut contraction = 0.0_f64;
    let mut native_magnitude = 0.0_f64;
    let mut reference_magnitude = 0.0_f64;
    let mut propagation = 0.0_f64;
    for dimension in 0..rank {
        let p = bf16::from_bits(predecessor[dimension]).to_f32() as f64;
        let s = bf16::from_bits(successor[dimension]).to_f32() as f64;
        let a = bf16::from_bits(native_hidden[dimension]).to_f32() as f64;
        let term = p * s * a;
        contraction += term;
        native_magnitude += term.abs();
        if let Some((_, hidden)) = reference {
            let e = hidden[dimension] as f64;
            reference_magnitude += (p * s * e).abs();
            propagation += (p * s).abs() * (a - e).abs();
        }
    }
    let rounding = 8.0 * rank as f64 * f32::EPSILON as f64;
    let contraction_bound = rounding * native_magnitude.max(1.0);
    let contraction_delta = (actual as f64 - contraction).abs();
    ensure!(contraction.is_finite() && contraction_delta.is_finite()
        && contraction_delta <= contraction_bound,
        "native edge contraction delta={contraction_delta} bound={contraction_bound}");
    // Every consumed native edge is checked, including native-only candidates.
    // Only ID-aligned edges have a reference-hidden propagation comparison.
    let Some((expected, _)) = reference else { return Ok(None); };
    let bound = propagation + rounding * (native_magnitude + reference_magnitude).max(1.0);
    let delta = (actual as f64 - expected as f64).abs();
    ensure!(bound.is_finite() && delta.is_finite() && delta <= bound,
        "reference edge delta={delta} bound={bound}");
    Ok(Some(delta))
}

fn compare_selector_oracles(
    device: &MlxDevice, registry: &mut KernelRegistry, runtime: &mut DFlash2Runtime,
    fixture: &SafeTensors<'_>, anchors: &[u32], case: &str,
) -> Result<()> {
    const K: usize = 16;
    let positions = runtime.config.proposal_tokens();
    let rank = runtime.config.selector_rank;
    let lanes = anchors.len();
    let expected_ids = i64_values(fixture.tensor("output.candidates")?).into_iter().map(|v| v as u32).collect::<Vec<_>>();
    let expected_unary = float_values(fixture.tensor("output.unary")?);
    let expected_edges = float_values(fixture.tensor("output.edges")?);
    let expected_hidden = float_values(fixture.tensor("output.selector_hidden")?);
    let expected_path = i64_values(fixture.tensor("output.path")?).into_iter().map(|v| v as u32).collect::<Vec<_>>();
    let derived_confidence = oracle_path_confidence(&expected_ids, &expected_unary, &expected_edges, &expected_path, lanes, positions)?;
    let old_confidence = float_values(fixture.tensor("output.confidence")?);
    ensure!(old_confidence.iter().all(|v| v.is_finite()) && max_abs(&old_confidence, &derived_confidence) <= CONFIDENCE_SAME_TABLE_MAX_ABS,
        "{case}: legacy confidence is not aligned to the pinned path; do not overwrite/relabel this fixture");
    let w = &runtime.workspace;
    let actual_ids = w.candidates.as_slice::<u32>()?;
    let actual_unary = w.unary.as_slice::<f32>()?;
    let actual_edges = w.edges.as_slice::<f32>()?;
    let actual_path = runtime.proposals()?;
    let own_confidence = oracle_path_confidence(actual_ids, actual_unary, actual_edges, actual_path, lanes, positions)?;
    ensure!(w.confidence.as_slice::<f32>()?.iter().all(|v| v.is_finite()),
        "{case}: nonfinite native production confidence");
    let confidence_delta = max_abs(w.confidence.as_slice::<f32>()?, &own_confidence);
    ensure!(confidence_delta <= CONFIDENCE_SAME_TABLE_MAX_ABS, "{case}: production confidence is not conditional on its selected predecessor: {confidence_delta}");
    let hidden = w.selector_hidden_bf16.as_slice::<u16>()?;
    let predecessor = runtime.tensors.predecessor_codebook.as_slice::<u16>()?;
    let successor = runtime.tensors.successor_codebook.as_slice::<u16>()?;
    let mut edge_count = 0;
    let mut worst_edge_delta = 0.0_f64;
    for lane in 0..lanes {
        for position in 0..positions {
            let row = lane * positions + position;
            for a_successor in 0..K {
                let token = actual_ids[row * K + a_successor];
                let e_successor = expected_ids[row * K..(row + 1) * K].iter().position(|id| *id == token);
                if let Some(index) = e_successor {
                    ensure!((actual_unary[row * K + a_successor] - expected_unary[row * K + index]).abs() <= 2.0,
                        "{case}: ID-aligned unary exceeds unchanged logits limit");
                }
                for a_predecessor in 0..if position == 0 { 1 } else { K } {
                    let pred_token = if position == 0 { anchors[lane] } else { actual_ids[(row - 1) * K + a_predecessor] };
                    let e_predecessor = if position == 0 { Some(0) } else {
                        expected_ids[(row - 1) * K..row * K].iter().position(|id| *id == pred_token)
                    };
                    let reference = e_predecessor.zip(e_successor).map(|(predecessor, successor)| (
                        expected_edges[(row * K + predecessor) * K + successor],
                        &expected_hidden[row * rank..(row + 1) * rank],
                    ));
                    let hidden_start = (lane * runtime.config.block_size + position + 1) * rank;
                    let predecessor_start = pred_token as usize * rank;
                    let successor_start = token as usize * rank;
                    let delta = oracle_aligned_edge(
                        actual_edges[(row * K + a_predecessor) * K + a_successor],
                        reference,
                        &hidden[hidden_start..hidden_start + rank],
                        &predecessor[predecessor_start..predecessor_start + rank],
                        &successor[successor_start..successor_start + rank],
                    ).with_context(|| format!("{case}: edge ({pred_token},{token}) lane={lane} position={position}"))?;
                    if let Some(delta) = delta {
                        edge_count += 1;
                        worst_edge_delta = worst_edge_delta.max(delta);
                    }
                }
            }
        }
    }
    // >=10 shared successors at row zero; >=10x10 shared ID pairs thereafter.
    ensure!(edge_count >= lanes * (10 + (positions - 1) * 100), "{case}: insufficient aligned edges");
    eprintln!("D1 {case} aligned_edges={edge_count} edge_max_abs={worst_edge_delta} native_own_table_confidence_max_abs={confidence_delta} end_to_end_confidence_max_abs={} (diagnostic: candidate/predecessor sets may differ)",
        max_abs(w.confidence.as_slice::<f32>()?, &derived_confidence));
    dispatch_oracle_score_tables(device, registry, runtime, &expected_ids, &expected_unary, &expected_edges, anchors)?;
    ensure!(runtime.proposals()? == expected_path.as_slice(), "{case}: native selector path differs on identical frozen score tables");
    ensure!(runtime.workspace.confidence.as_slice::<f32>()?.iter().all(|v| v.is_finite()),
        "{case}: nonfinite native frozen-table confidence");
    let delta = max_abs(runtime.workspace.confidence.as_slice::<f32>()?, &derived_confidence);
    ensure!(delta <= CONFIDENCE_SAME_TABLE_MAX_ABS, "{case}: native frozen-table confidence max_abs={delta}");
    eprintln!("D1 {case} frozen_table_confidence_max_abs={delta}");
    Ok(())
}

#[test]
fn reference_confidence_uses_token_aligned_predecessors_not_column_order() {
    let mut ids = (0..32).collect::<Vec<u32>>();
    ids.swap(0, 5);
    let mut unary = vec![0.0; 32];
    unary[5] = 2.0;
    let mut edges = vec![0.0; 512];
    edges[(16 + 5) * 16 + 9] = 4.0;
    let confidence = oracle_path_confidence(&ids, &unary, &edges, &[0, 25], 1, 2).unwrap();
    assert!((confidence[0] - 1.0 / (1.0 + 15.0 * (-2.0_f32).exp())).abs() < 1e-6);
    assert!((confidence[1] - 1.0 / (1.0 + 15.0 * (-4.0_f32).exp())).abs() < 1e-6);
    assert!(oracle_path_confidence(&ids, &unary, &edges, &[0, 16], 1, 2).is_err());
}

#[test]
fn reference_confidence_rounds_score_addition_before_softmax() {
    let candidates = (0..16).collect::<Vec<u32>>();
    let unary = vec![8192.0; 16];
    let mut edges = vec![0.0; 256];
    edges[1] = 0.0004;
    let confidence = oracle_path_confidence(&candidates, &unary, &edges, &[0], 1, 1).unwrap();
    assert_eq!(confidence, vec![0.0625], "all sixteen F32 scores tie exactly");
}

#[test]
fn aligned_edge_rejects_wrong_sign_despite_hidden_propagation_allowance() {
    let ones = vec![bf16::from_f32(1.0).to_bits(); 256];
    let reference = vec![0.0; 256];
    assert_eq!(oracle_aligned_edge(256.0, Some((0.0, &reference)), &ones, &ones, &ones).unwrap(), Some(256.0));
    assert!(oracle_aligned_edge(-256.0, Some((0.0, &reference)), &ones, &ones, &ones).is_err());
    assert!(oracle_aligned_edge(-256.0, None, &ones, &ones, &ones).is_err(),
        "a native-only candidate cannot bypass its own contraction check");
}
