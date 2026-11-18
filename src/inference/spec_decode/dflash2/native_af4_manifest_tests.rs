// Native-AF4 oracle: independent of the original BF16 oracle and its bounds.
// All descriptors are verified before loading a model. No fixture is optional once configured.
const AF4_PEER: &str = "031a25a6125af2d7acf7096580afe56eb33d65bc";
const AF4_SOURCE: &str = "614781b037296ef0708cdc3286beb68e19322d78";
const AF4_PRODUCER: &str = "faaf6e739c24b301bc0d8ac07809734260897a0704b843bae9ced5c786791456";
const AF4_METAL: &str = "7d804ede552a29f3d3bfb52463735f3d6a08f63cb840a96b7a56edf6cbb01409";
const AF4_REFERENCE_REVISION: &str = "lossless-captured-hidden-rebuild-v1";

fn af4_usize(value: &serde_json::Value) -> Result<usize> {
    usize::try_from(value.as_u64().context("missing unsigned fixture field")?).map_err(Into::into)
}

fn af4_digest(bytes: &[u8]) -> String { hex::encode(Sha256::digest(bytes)) }

fn af4_descriptor(base: &Path, record: &serde_json::Value, relative: bool) -> Result<PathBuf> {
    let name = record["path"].as_str().context("missing descriptor path")?;
    let path = Path::new(name);
    if relative {
        ensure!(!path.is_absolute() && path.components().all(|part| matches!(part, std::path::Component::Normal(_))),
            "evidence path must be confined: {name}");
    }
    let path = if relative { base.join(path) } else { path.to_path_buf() };
    ensure!(std::fs::metadata(&path)?.is_file(), "descriptor is not a regular file: {}", path.display());
    ensure!(std::fs::metadata(&path)?.len() == record["bytes"].as_u64().context("descriptor bytes missing")?,
        "descriptor length differs: {}", path.display());
    ensure!(record["sha256"].as_str() == Some(oracle_file_sha256(&path)?.as_str()),
        "descriptor SHA differs: {}", path.display());
    if relative {
        ensure!(path.canonicalize()?.starts_with(base.canonicalize()?), "evidence symlink escapes bundle");
    }
    Ok(path)
}

fn af4_json(path: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn af4_frozen_sha(family: &str, case: &str) -> Result<&'static str> {
    Ok(match (family, case) {
        ("qwen36", "short") => "35620a39f4f71d078e5a38584f103cc8efc522b71b47b8885b859d73510b5bac",
        ("qwen36", "wrap") => "3f2bf6cf147da87b365481d096652b337a254518d7d6072a10c5116648307204",
        ("qwen36", "single") => "4a5c07468d006f9e6e513033433150dcfadfc907e0d6f3b09aa4d159ed85e2d5",
        ("qwen36", "multi") => "bf2bdec24a726d9085defeff87415d26e4541c06fb770a2e35bb248eb857e799",
        ("qwen36", "restore") => "470ea68f46b276e27f36436c20930e7bf9a71861fa34a0b36bb74af420d79408",
        ("qwen38", "short") => "66466e1c7e5003162cd23140f5985a79d6be422b5900e7fa42571473e06316c3",
        ("qwen38", "wrap") => "9cc5b2a3d76af568d63a7a889ce590cc345829c8c061f425dcf34738faa70c7a",
        ("qwen38", "single") => "e280da2eb2a93154aa2b2d03e08dde7c84b7368f8f42ea86d39b4f89f2c6dfb0",
        ("qwen38", "multi") => "cca1a04506c22b20344958018454e371d72b2e157b2c5cbeea10a1b81ce2839b",
        ("qwen38", "restore") => "952f7ef79ebf81d323ccc9609ee17317729eac6107b160fda2bdde2772920c58",
        _ => bail!("unsupported native-AF4 family/case"),
    })
}

fn af4_tensor<'a>(fixture: &SafeTensors<'a>, name: &str, dtype: Dtype, shape: &[usize]) -> Result<TensorView<'a>> {
    let view = fixture.tensor(name).with_context(|| format!("missing native-AF4 tensor {name}"))?;
    ensure!(view.dtype() == dtype && view.shape() == shape, "{name}: dtype/shape differs: {:?} {:?}", view.dtype(), view.shape());
    if matches!(dtype, Dtype::F32 | Dtype::BF16) {
        ensure!(float_values(view.clone()).iter().all(|x| x.is_finite()), "{name}: nonfinite semantic tensor");
    }
    Ok(view)
}

fn af4_verify_bundle(manifest: &serde_json::Value, base: &Path) -> Result<Vec<u8>> {
    ensure!(manifest["schema"] == "hf2q.dflash2-native-af4-fixture.v2" && manifest["oracle_kind"] == "pinned-native-af4", "native-AF4 schema/oracle differs");
    af4_require_reference_policy(manifest)?;
    let family = manifest["family"].as_str().context("family missing")?;
    let case = manifest["case"].as_str().context("case missing")?;
    let b = &manifest["bindings"];
    ensure!(b["peer_commit"] == AF4_PEER && b["hf2q_source"] == AF4_SOURCE && b["producer_sha256"] == AF4_PRODUCER
        && b["metallib_sha256"] == AF4_METAL, "native-AF4 pinned source differs");
    ensure!(b["original_fixture"]["sha256"] == af4_frozen_sha(family, case)?, "original frozen fixture identity differs");
    let evidence = manifest["evidence"].as_object().context("evidence missing")?;
    let mut records = std::collections::BTreeMap::new();
    for key in ["inputs", "transparency", "plain_manifest", "observed_manifest", "weights", "build_provenance",
        "completion", "plain_dispatch", "observed_dispatch", "checkpoint_config", "original_fixture_manifest", "capture_manifest"] {
        let path = af4_descriptor(base, evidence.get(key).with_context(|| format!("missing evidence {key}"))?, true)?;
        records.insert(key, af4_json(&path)?);
    }
    // Verify extra evidence descriptors too; unknown additions cannot hide an unverified dependency.
    for record in evidence.values() { af4_descriptor(base, record, true)?; }
    for key in ["export.py", "dflash2_native_af4_evidence.py", "dflash2_native_af4_io.py", "source-inventory.json"] {
        ensure!(evidence.contains_key(key), "matched-reference exporter dependency missing: {key}");
    }
    ensure!(b["exporter_source_sha256"].as_str().is_some()
        && b["exporter_source_sha256"] == evidence["export.py"]["sha256"], "matched-reference exporter binding differs");
    let build = &records["build_provenance"];
    ensure!(build["schema"] == "hf2q.peer-native-af4-build.v2" && build["peer_commit"] == AF4_PEER,
        "native build source schema differs");
    ensure!(build["metallib"]["sha256"] == AF4_METAL && b["native_build_provenance_sha256"] == evidence["build_provenance"]["sha256"], "native build/metallib binding differs");
    for mode in ["plain", "observed"] {
        ensure!(b[format!("{mode}_executable_sha256")] == build["binaries"][mode]["sha256"], "{mode} executable binding differs");
        af4_descriptor(base, &build["binaries"][mode], false)?;
    }
    af4_descriptor(base, &build["metallib"], false)?;
    ensure!(build["translation_units"].as_array().context("translation units missing")?.len() == 90,
        "native build must retain all 90 production translation units");
    for key in ["source_inputs", "objects", "harness_inputs", "frozen_source_inputs", "frozen_harness_inputs", "link_maps"] {
        let entries = if let Some(entries) = build[key].as_array() { entries.iter().collect::<Vec<_>>() }
            else { build[key].as_object().with_context(|| format!("build {key} missing"))?.values().collect::<Vec<_>>() };
        ensure!(!entries.is_empty(), "empty build {key}");
        for record in entries { af4_descriptor(base, record, false)?; }
    }
    af4_verify_source_closure(base, build)?;
    let inputs = &records["inputs"];
    ensure!(inputs["schema"] == "hf2q.peer-native-draft-inputs.v1" && inputs["build"]["sha256"] == b["native_build_provenance_sha256"], "input build provenance differs");
    ensure!(inputs["build_contents"] == *build && inputs["fixture"] == b["original_fixture"], "input provenance differs");
    ensure!(inputs["fixture_metadata"] == b["original_fixture_metadata"], "original metadata not retained verbatim");
    let original_path = af4_descriptor(base, &b["original_fixture"], false)?;
    let original_bytes = std::fs::read(&original_path)?;
    let original = SafeTensors::deserialize(&original_bytes)?;
    let path = af4_descriptor(base, &manifest["fixture"], true)?;
    let bytes = std::fs::read(path)?;
    let fixture = SafeTensors::deserialize(&bytes)?;
    let (_, header) = SafeTensors::read_metadata(&bytes)?;
    let metadata = header.metadata().as_ref().context("native-AF4 tensor metadata missing")?;
    for (key, expected) in [("schema", "hf2q.dflash2-native-af4-fixture.v2"), ("oracle_kind", "pinned-native-af4"),
        ("family", family), ("case", case), ("peer_commit", AF4_PEER), ("reference_revision", AF4_REFERENCE_REVISION)] {
        ensure!(metadata.get(key).map(String::as_str) == Some(expected), "tensor metadata {key} differs");
    }
    let old_names = original.names().into_iter().filter(|name| name.starts_with("input.")).collect::<std::collections::BTreeSet<_>>();
    let names = fixture.names().into_iter().filter(|name| name.starts_with("input.")).collect::<std::collections::BTreeSet<_>>();
    ensure!(old_names == names, "original input coverage differs");
    for name in ["input.noise_embedding", "input.target_hidden", "input.initial_target_hidden"] {
        if name == "input.initial_target_hidden" && case != "restore" { continue; }
        let semantic = original.tensor(name)?;
        ensure!(semantic.dtype() == Dtype::F32, "original semantic input is not F32");
        let rounded = af4_tensor(&fixture, &format!("native.{name}"), Dtype::BF16, semantic.shape())?;
        let values = float_values(semantic);
        ensure!(values.iter().all(|x| x.is_finite()), "original input is nonfinite");
        let expected = values.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect::<Vec<_>>();
        ensure!(rounded.data() == expected.as_slice(), "native input was not the original F32 semantic input rounded once: {name}");
    }
    for name in old_names {
        let a = fixture.tensor(name)?;
        let e = original.tensor(name)?;
        ensure!(a.dtype() == e.dtype() && a.shape() == e.shape() && a.data() == e.data(), "original semantic input changed: {name}");
    }
    let plain = &records["plain_manifest"];
    let observed = &records["observed_manifest"];
    for (mode, value) in [("plain", plain), ("observed", observed)] {
        ensure!(value["schema"] == "hf2q.peer-native-draft-tensors.v2" && value["mode"] == mode && value["family"] == family,
            "{mode} native capture schema differs");
        af4_require_reference_policy(value)?;
        ensure!(value["fixture_metadata"] == b["original_fixture_metadata"] && value["geometry"] == manifest["geometry"], "{mode} capture geometry/source differs");
    }
    af4_verify_tensor_inventory(manifest, &fixture, plain, observed, &records["transparency"])?;
    af4_verify_lossless_restore(manifest, &fixture)?;
    af4_verify_execution(base, manifest, &records)?;
    af4_verify_edge_layout(manifest, &fixture, observed)?;
    ensure!(plain["state"] == observed["state"], "observer changed actual state metadata");
    for key in ["absolute_position", "cursor", "valid_rows", "valid_ring_intervals", "row_absolute_positions",
        "restored", "restore_initial_rows", "restore_kind", "restore_input_tensors", "saved_window_tensors"] {
        ensure!(!observed["state"][key].is_null() && observed["state"][key] == manifest["state"][key],
            "exported state does not match actual native {key}");
    }
    ensure!(observed["state"]["context_lengths"] == serde_json::json!(vec![af4_usize(&manifest["state"]["absolute_position"])?; af4_usize(&manifest["geometry"]["lanes"])?]),
        "actual native context lengths differ");
    let ordinary = plain["tensors"].as_object().context("plain tensor inventory missing")?;
    ensure!(!ordinary.is_empty(), "plain inventory empty");
    for (name, record) in ordinary {
        let other = &observed["tensors"][name];
        for key in ["sha256", "bytes", "dtype", "shape"] {
            ensure!(!record[key].is_null() && record[key] == other[key], "observation changed ordinary {name}.{key}");
        }
    }
    ensure!(records["transparency"]["schema"] == "hf2q.peer-native-draft-transparency.v1", "transparency schema differs");
    // Hash retained bytes, not a transparency boolean, against every raw capture descriptor.
    for (name, record) in observed["tensors"].as_object().context("observed tensor inventory missing")? {
        let stored = if name == "output.edges" { "native.edges.storage" } else { name };
        let view = fixture.tensor(stored).with_context(|| format!("raw observed tensor missing: {stored}"))?;
        ensure!(record["shape"] == serde_json::to_value(view.shape())?, "raw capture shape changed: {name}");
        if name != "output.candidates" && name != "output.path" {
            ensure!(record["dtype"] == format!("{:?}", view.dtype()), "raw capture dtype changed: {name}");
        }
        let raw: std::borrow::Cow<'_, [u8]> = if (name == "output.candidates" || name == "output.path") && record["dtype"] == "U32" {
            ensure!(view.dtype() == Dtype::I64, "lossless ID widening missing");
            std::borrow::Cow::Owned(i64_values(view.clone()).into_iter().map(u32::try_from).collect::<Result<Vec<_>, _>>()?
                .into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())
        } else { std::borrow::Cow::Borrowed(view.data()) };
        ensure!(af4_digest(&raw) == record["sha256"].as_str().context("raw SHA missing")?
            && raw.len() == af4_usize(&record["bytes"])?, "raw capture bytes changed: {name}");
    }
    ensure!(manifest["weights"] == records["weights"], "embedded weight inventory differs");
    let original_manifest = &records["original_fixture_manifest"];
    ensure!(original_manifest["hf2q_source"] == AF4_SOURCE && original_manifest["producer_sha256"] == AF4_PRODUCER,
        "original producer/source binding differs");
    let file_name = original_path.file_name().and_then(|x| x.to_str()).context("original fixture name missing")?;
    let descriptor = if family == "qwen36" { &QWEN36_DFLASH2 } else { &QWEN38_DFLASH2 };
    bind_reference_fixture(descriptor, family, case, file_name, &original_bytes,
        b["target_gguf_sha256"].as_str().context("target binding missing")?, original_manifest)?;
    Ok(bytes)
}

fn af4_geometry(manifest: &serde_json::Value, config: &DFlash2Config) -> Result<(usize, usize, usize)> {
    let case = manifest["case"].as_str().context("case missing")?;
    let (lanes, rows, start, offset) = match case {
        "short" => (1, 3, 0, 0), "wrap" => (1, 2048, 0, 0), "single" => (1, 17, 0, 1),
        "multi" => (2, 17, 0, 0), "restore" => (1, 3, 12, 0), _ => bail!("unsupported case"),
    };
    let g = &manifest["geometry"];
    for (key, value) in [("lanes", lanes), ("context_rows", rows), ("context_start", start),
        ("layers", config.num_hidden_layers), ("hidden_size", config.hidden_size), ("vocabulary_size", config.vocab_size),
        ("target_hidden_size", config.fc_input_dim()), ("dynamic_size", config.convolution_groups() * 4),
        ("kv_heads", config.num_key_value_heads), ("head_dimension", config.head_dim), ("query_rows", config.block_size),
        ("proposal_rows", config.proposal_tokens()), ("draft_window", config.sliding_window),
        ("intermediate_size", config.intermediate_size), ("selector_rank", config.selector_rank),
        ("attention_size", config.num_attention_heads * config.head_dim),
        ("qkv_size", (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim)] {
        ensure!(af4_usize(&g[key])? == value, "native geometry {key} differs");
    }
    ensure!(g["lane_offset"] == offset.to_string() && g["restore"] == (case == "restore"), "original lane/restore geometry differs");
    ensure!(g["target_layer_ids"] == serde_json::to_value(&config.target_layer_ids)? && g["rotary_theta"].as_f64() == Some(config.rope_theta), "target layers/RoPE differ");
    ensure!(config.block_size == 8 && config.proposal_tokens() == 7 && config.selector_top_k == 16
        && config.sliding_window == 2048 && config.head_dim == 128, "original fixed geometry differs");
    Ok((lanes, rows, start))
}

fn af4_verify_source_closure(base: &Path, build: &serde_json::Value) -> Result<()> {
    for (original, frozen) in [("source_inputs", "frozen_source_inputs"), ("harness_inputs", "frozen_harness_inputs")] {
        let originals = build[original].as_array().context("source closure missing")?;
        let copies = build[frozen].as_array().context("frozen source closure missing")?;
        ensure!(originals.len() == copies.len(), "frozen source closure count differs");
        for (a, b) in originals.iter().zip(copies) {
            ensure!(a["bytes"] == b["bytes"] && a["sha256"] == b["sha256"], "frozen source closure differs");
        }
    }
    let units = build["translation_units"].as_array().context("translation units missing")?;
    let names = units.iter().map(|x| x.as_str().context("invalid production TU name")).collect::<Result<std::collections::BTreeSet<_>>>()?;
    ensure!(names.len() == 90 && names.contains("runtime/model/DFlashDraft.cpp"), "incomplete production TU closure");
    af4_verify_pinned_build(base, build)?;
    let shader_path = af4_descriptor(base, &build["shader_provenance"], false)?;
    let shader = af4_json(&shader_path)?;
    ensure!(shader == build["shader_provenance_contents"] && shader["peer_commit"] == AF4_PEER
        && shader["mode"] == "source" && shader["metallib"]["sha256"] == AF4_METAL, "shader source provenance differs");
    for record in shader["source_inputs"].as_array().context("shader source closure missing")? {
        af4_descriptor(base, record, false)?;
    }
    let source = build["frozen_source_inputs"].as_array().context("frozen source closure missing")?.iter()
        .find(|x| x["path"].as_str().is_some_and(|p| p.ends_with("/runtime/model/DFlashDraft.cpp")))
        .context("original production draft TU missing")?;
    let mut original = std::fs::read_to_string(af4_descriptor(base, source, false)?)?;
    for addition in build["overlay_additions"].as_array().context("observer overlay additions missing")? {
        let (anchor, replacement) = if let Some(anchor) = addition["anchor"].as_str() {
            (anchor, format!("{anchor}{}", addition["addition"].as_str().context("observer addition missing")?))
        } else {
            let anchor = addition["before"].as_str().context("observer before anchor missing")?;
            (anchor, format!("{}{anchor}", addition["addition"].as_str().context("observer addition missing")?))
        };
        ensure!(original.matches(anchor).count() == 1, "observer overlay anchor ambiguous");
        original = original.replacen(anchor, &replacement, 1);
    }
    let overlay = std::fs::read_to_string(af4_descriptor(base, &build["overlay_source"], false)?)?;
    ensure!(original == overlay, "observation changed production source beyond the add-only overlay");
    Ok(())
}

fn af4_verify_execution(base: &Path, manifest: &serde_json::Value,
    records: &std::collections::BTreeMap<&str, serde_json::Value>) -> Result<()> {
    let inputs = &records["inputs"];
    let build = &records["build_provenance"];
    let completion = &records["completion"];
    let models = inputs["model_inputs"].as_array().context("native model inputs missing")?;
    ensure!(models.len() == 3, "native actual model coverage differs");
    for record in models { af4_descriptor(base, record, false)?; }
    let receipts = completion["receipts"].as_array().context("native completion receipts missing")?;
    ensure!(receipts.len() == 2 && completion["status"] == "captured", "native run did not complete both processes");
    for mode in ["plain", "observed"] {
        let receipt = &records[format!("{mode}_dispatch").as_str()];
        let argv = receipt["argv"].as_array().context("native dispatch argv missing")?;
        ensure!(argv.len() == 8 && receipt["variant"] == mode && receipt["returncode"] == 0
            && receipt["monotonic_end_ns"].as_u64().context("native completion time missing")?
                >= receipt["monotonic_start_ns"].as_u64().context("native start time missing")?
            && receipts.contains(receipt), "native {mode} execution receipt differs");
        ensure!(argv[0] == build["binaries"][mode]["path"] && argv[1] == manifest["family"]
            && argv[2] == inputs["fixture"]["path"] && argv[4] == models[0]["path"]
            && argv[5] == build["metallib"]["path"] && argv[7] == mode, "native {mode} dispatch source/model differs");
        ensure!(Path::new(argv[3].as_str().context("native checkpoint path missing")?)
            == Path::new(models[1]["path"].as_str().context("native config path missing")?).parent().context("native config parent missing")?,
            "native checkpoint dispatch differs");
        let weights_path = Path::new(argv[6].as_str().context("native output path missing")?).join("weights.json");
        ensure!(af4_json(&weights_path)? == manifest["weights"], "actual {mode} prepared inventory differs");
    }
    let identities = models.iter().map(|r| r["sha256"].as_str().context("model SHA missing")).collect::<Result<std::collections::BTreeSet<_>>>()?;
    let b = &manifest["bindings"];
    ensure!(identities == [b["target_gguf_sha256"].as_str().context("target SHA missing")?,
        b["checkpoint_config_sha256"].as_str().context("config SHA missing")?,
        b["checkpoint_weights_sha256"].as_str().context("weights SHA missing")?].into_iter().collect(), "native model identity closure differs");
    Ok(())
}

fn af4_verify_edge_layout(manifest: &serde_json::Value, fixture: &SafeTensors<'_>, observed: &serde_json::Value) -> Result<()> {
    let lanes = af4_usize(&manifest["geometry"]["lanes"])?;
    let edge = &observed["tensors"]["output.edges"];
    for key in ["axes", "valid_regions", "inactive_region", "region_interval_convention"] {
        ensure!(!edge[key].is_null() && edge[key] == manifest["edge_validity"][key], "native edge validity metadata differs: {key}");
    }
    ensure!(edge["valid_regions"] == serde_json::json!([
        {"lane":[0,lanes],"proposal_position":[0,1],"predecessor_candidate":[0,1],"successor_candidate":[0,16]},
        {"lane":[0,lanes],"proposal_position":[1,7],"predecessor_candidate":[0,16],"successor_candidate":[0,16]}])
        && edge["inactive_region"]["proposal_position"] == serde_json::json!([0,1])
        && edge["inactive_region"]["predecessor_candidate"] == serde_json::json!([1,16]), "native edge validity coverage differs");
    let raw = fixture.tensor("native.edges.storage")?;
    let canonical = af4_tensor(fixture, "output.edges", Dtype::F32, &[lanes,7,16,16])?;
    ensure!(raw.dtype() == Dtype::F32 && raw.shape() == [lanes,7,16,16], "native edge raw storage shape differs");
    for lane in 0..lanes {
        let base = lane * 7 * 16 * 16 * 4;
        for predecessor in 0..16 {
            ensure!(canonical.data()[base + predecessor * 64..base + (predecessor + 1) * 64]
                == raw.data()[base..base + 64], "anchor edge broadcast changed actual native scores");
        }
        ensure!(canonical.data()[base + 1024..base + 7 * 1024] == raw.data()[base + 1024..base + 7 * 1024],
            "native valid non-anchor edges changed");
    }
    Ok(())
}

fn af4_finite_payload(view: &TensorView<'_>, name: &str) -> Result<()> {
    let finite = match view.dtype() {
        Dtype::BF16 => view.data().chunks_exact(2).all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()),
        Dtype::F32 => view.data().chunks_exact(4).all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()),
        _ => true,
    };
    ensure!(finite, "retained raw float tensor contains nonfinite data: {name}");
    Ok(())
}

fn af4_verify_tensor_inventory(manifest: &serde_json::Value, fixture: &SafeTensors<'_>,
    plain: &serde_json::Value, observed: &serde_json::Value, transparency: &serde_json::Value) -> Result<()> {
    let g = &manifest["geometry"];
    let lanes = af4_usize(&g["lanes"])?;
    let hidden = af4_usize(&g["hidden_size"])?;
    let vocabulary = af4_usize(&g["vocabulary_size"])?;
    let rank = af4_usize(&g["selector_rank"])?;
    let layers = af4_usize(&g["layers"])?;
    let mut expected = std::collections::BTreeMap::<String, (&str, Vec<usize>)>::new();
    for (name, dtype, shape) in [
        ("output.final_hidden", "BF16", vec![lanes,8,hidden]),
        ("output.logits", "F32", vec![lanes,7,vocabulary]),
        ("output.selector_hidden", "BF16", vec![lanes,7,rank]),
        ("output.candidates", "U32", vec![lanes,7,16]), ("output.path", "U32", vec![lanes,7]),
        ("output.unary", "F32", vec![lanes,7,16]), ("output.confidence", "F32", vec![lanes,7]),
        ("output.edges", "F32", vec![lanes,7,16,16]),
        ("native.logits.all_rows", "F32", vec![lanes,8,vocabulary]),
        ("native.selector_hidden.all_rows", "BF16", vec![lanes,8,rank]),
        ("native.selector.partial_ids", "U32", vec![lanes*7*16*8]),
        ("native.selector.partial_values_and_edges", "F32", vec![lanes*7*16*(8+16)]),
        ("input.anchor", "I64", vec![lanes]),
        ("native.input.noise_embedding", "BF16", vec![lanes,8,hidden]),
        ("native.input.target_hidden", "BF16", vec![lanes,af4_usize(&g["context_rows"])?,af4_usize(&g["target_hidden_size"])?]),
    ] { expected.insert(name.to_string(), (dtype, shape)); }
    let restoring = manifest["case"] == "restore";
    if restoring {
        expected.insert("native.input.initial_target_hidden".to_string(),
            ("BF16", vec![lanes,12,af4_usize(&g["target_hidden_size"])?]));
        expected.insert("native.restore.input.initial_target_hidden".to_string(),
            ("BF16", vec![lanes,12,af4_usize(&g["target_hidden_size"])?]));
    }
    for layer in 0..layers {
        expected.insert(format!("output.layer.{layer}.output"), ("BF16", vec![lanes,8,hidden]));
        for branch in ["attention", "mlp"] {
            for boundary in ["prepare", "finish.fused", "finish.incoming", "dynamic.full"] {
                let width = if boundary == "dynamic.full" { af4_usize(&g["dynamic_size"])? } else { hidden };
                expected.insert(format!("output.layer.{layer}.{branch}.{boundary}"), ("BF16", vec![lanes,8,width]));
            }
        }
        for lane in 0..lanes {
            expected.insert(format!("native.state.lane{lane}.layer{layer}.keys"), ("BF16", vec![8,2048,128]));
            expected.insert(format!("native.state.lane{lane}.layer{layer}.values"), ("BF16", vec![8,128,2048]));
            if restoring {
                for phase in ["before", "after"] {
                    expected.insert(format!("native.restore.lane{lane}.{phase}.layer{layer}.keys"), ("BF16", vec![8,2048,128]));
                    expected.insert(format!("native.restore.lane{lane}.{phase}.layer{layer}.values"), ("BF16", vec![8,128,2048]));
                }
            }
        }
    }
    for lane in 0..lanes {
        let size = 2048 * (hidden / 2 + hidden / 64 * 4);
        expected.insert(format!("native.state.lane{lane}.window"), ("U8", vec![size]));
    }
    let descriptors = observed["tensors"].as_object().context("observed inventory missing")?;
    for (input, rows) in [("input.target_hidden", af4_usize(&g["context_rows"])?),
        ("input.initial_target_hidden", if restoring { 12 } else { 0 }),
        ("restore.input.initial_target_hidden", if restoring { 12 } else { 0 })] {
        if rows == 0 { continue; }
        for lane in 0..lanes {
            let prefix = format!("native.prefill.{input}.lane{lane}.chunk");
            let mut chunks = Vec::new();
            for name in descriptors.keys() {
                if let Some(offset) = name.strip_prefix(&prefix).and_then(|s| s.strip_suffix(".native.context.projected")) {
                    chunks.push((offset.parse::<usize>()?, name));
                }
            }
            chunks.sort_unstable_by_key(|(offset, _)| *offset);
            ensure!(!chunks.is_empty(), "observed context descriptor coverage missing");
            let mut cursor = 0;
            for (offset, name) in chunks {
                let shape = descriptors[name]["shape"].as_array().context("context descriptor shape missing")?;
                ensure!(shape.len() == 2 && af4_usize(&shape[1])? == hidden, "context descriptor width differs");
                let count = af4_usize(&shape[0])?;
                ensure!(offset == cursor && count > 0 && count <= rows - cursor, "context descriptor rows gapped/excessive");
                let shape = vec![count,hidden];
                expected.insert(name.clone(), ("BF16", shape.clone()));
                let paired = name.strip_suffix("projected").context("context name suffix missing")?.to_string() + "hidden";
                expected.insert(paired, ("BF16", shape));
                cursor += count;
            }
            ensure!(cursor == rows, "observed context descriptor rows incomplete");
        }
    }
    ensure!(descriptors.keys().cloned().collect::<std::collections::BTreeSet<_>>()
        == expected.keys().cloned().collect(), "complete observed raw tensor inventory differs");
    for (name, (dtype, shape)) in &expected {
        ensure!(descriptors[name]["dtype"] == *dtype && descriptors[name]["shape"] == serde_json::to_value(shape)?,
            "observed descriptor dtype/shape differs: {name}");
    }
    let ordinary = expected.keys().filter(|name| !name.starts_with("output.layer.") && !name.starts_with("native.prefill."))
        .cloned().collect::<std::collections::BTreeSet<_>>();
    let plain = plain["tensors"].as_object().context("plain inventory missing")?;
    let transparent = transparency["per_tensor_exact_bytes_equal"].as_object().context("transparency inventory missing")?;
    ensure!(plain.keys().cloned().collect::<std::collections::BTreeSet<_>>() == ordinary
        && transparent.keys().cloned().collect::<std::collections::BTreeSet<_>>() == ordinary
        && transparent.values().all(|x| x == true), "plain/observed/transparency ordinary inventory differs");
    // Include every retained float payload, not just proposal rows or descriptors visited later.
    for (name, view) in fixture.iter() { af4_finite_payload(&view, name)?; }
    Ok(())
}
