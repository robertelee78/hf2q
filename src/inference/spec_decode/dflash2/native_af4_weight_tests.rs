// Decode the actual prepared HF planes. Never quantize checkpoint weights a second time.
fn af4_logical_hashes(projection: &super::super::tensors::DFlash2Projection,
    rows: usize, columns: usize, first: usize, count: usize) -> Result<[String; 3]> {
    use super::super::tensors::DFlash2Projection;
    let DFlash2Projection::Af4G64 { weights, metadata } = projection else { bail!("native-AF4 requires the real AF4 production route"); };
    ensure!(rows % 64 == 0 && columns % 64 == 0 && first + count <= rows, "invalid prepared AF4 geometry");
    let words = weights.as_slice::<u32>()?;
    let meta = metadata.as_slice::<u32>()?;
    ensure!(words.len() == rows * columns / 8 && meta.len() == rows * columns / 64, "prepared AF4 plane extent differs");
    let mut codes = Sha256::new();
    let mut scales = Sha256::new();
    let mut biases = Sha256::new();
    let mut row_codes = vec![0_u8; columns];
    let mut row_scales = vec![0_u8; columns / 32];
    let mut row_biases = vec![0_u8; columns / 32];
    for row in first..first + count {
        for group in 0..columns / 64 {
            let at = (row / 64 * (columns / 64) + group) * 64 + row % 64;
            let value = meta[at];
            let scale = value as u16;
            let bias = (value >> 16) as u16;
            ensure!(bf16::from_bits(scale).is_finite() && bf16::from_bits(bias).is_finite(), "nonfinite AF4 metadata");
            row_scales[group * 2..group * 2 + 2].copy_from_slice(&scale.to_le_bytes());
            row_biases[group * 2..group * 2 + 2].copy_from_slice(&bias.to_le_bytes());
        }
        for group in 0..columns / 32 {
            for chunk in 0..4 {
                let at = ((row / 64 * (columns / 32) + group) * 64 + row % 64) * 4 + chunk;
                let word = words[at];
                for pair in 0..4 {
                    let first_column = group * 32 + 16 * (pair >> 1) + 4 * chunk + 2 * (pair & 1);
                    row_codes[first_column] = ((word >> (4 * pair)) & 15) as u8;
                    row_codes[first_column + 1] = ((word >> (16 + 4 * pair)) & 15) as u8;
                }
            }
        }
        codes.update(&row_codes); scales.update(&row_scales); biases.update(&row_biases);
    }
    Ok([hex::encode(codes.finalize()), hex::encode(scales.finalize()), hex::encode(biases.finalize())])
}

fn af4_check_projection(record: &serde_json::Value, projection: &super::super::tensors::DFlash2Projection,
    rows: usize, columns: usize, first: usize, count: usize, name: &str) -> Result<()> {
    ensure!(af4_usize(&record["input_size"])? == columns && af4_usize(&record["output_size"])? == count, "{name}: native projection shape differs");
    ensure!(record["destination"] == "BF16" && af4_usize(&record["rotation_signs"]["bytes"])? == 0, "{name}: projection precision/rotation differs");
    let segments = record["segments"].as_array().context("AF4 segments missing")?;
    ensure!(segments.len() == 1, "{name}: unsupported/missing actual AF4 segment");
    let segment = &segments[0];
    ensure!(segment["format"] == "af4g64" && segment["format_id"] == 26 && segment["storage_layout"] == "planes"
        && af4_usize(&segment["column_offset"])? == 0 && af4_usize(&segment["input_size"])? == columns
        && af4_usize(&segment["output_size"])? == count, "{name}: AF4 segment geometry differs");
    ensure!(af4_usize(&segment["plane0"]["bytes"])? == count * columns / 2
        && af4_usize(&segment["metadata"]["bytes"])? == count * columns / 16
        && af4_usize(&segment["plane1"]["bytes"])? == 0, "{name}: physical AF4 plane extent differs");
    for key in ["plane0", "metadata"] {
        ensure!(af4_usize(&segment[key]["bytes"])? > 0 && segment[key]["sha256"].as_str().is_some_and(|s| s.len() == 64), "{name}: missing physical {key} inventory");
    }
    let logical = &segment["logical_af4"];
    ensure!(logical["source"] == "actual-loaded-prepared-planes" && logical["group_size"] == 64,
        "{name}: logical inventory was not obtained from actual prepared planes");
    ensure!(logical["codes_shape"] == serde_json::json!([count, columns])
        && logical["scale_bias_shape"] == serde_json::json!([count, columns / 64])
        && af4_usize(&logical["codes_count"])? == count * columns
        && af4_usize(&logical["scale_bias_count"])? == count * columns / 64, "{name}: logical shape/count differs");
    let hashes = af4_logical_hashes(projection, rows, columns, first, count)?;
    for (key, actual) in ["codes_sha256", "scale_bf16_sha256", "bias_bf16_sha256"].into_iter().zip(hashes) {
        ensure!(logical[key].as_str() == Some(actual.as_str()), "{name}: actual prepared {key} differs");
    }
    eprintln!("NATIVE_AF4_WEIGHT {name}: actual prepared logical planes match");
    Ok(())
}

fn af4_check_buffer(record: &serde_json::Value, buffer: &MlxBuffer, norm: bool, name: &str) -> Result<()> {
    let bytes = if norm {
        buffer.as_slice::<f32>()?.iter().map(|&x| {
            let rounded = bf16::from_f32(x);
            ensure!(x.is_finite() && rounded.to_f32().to_bits() == x.to_bits(), "{name}: norm is not stored BF16 widened exactly");
            Ok(rounded.to_le_bytes())
        }).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect::<Vec<_>>()
    } else {
        let values = buffer.as_slice::<u16>()?;
        ensure!(values.iter().all(|&x| bf16::from_bits(x).is_finite()), "{name}: nonfinite stored BF16 weight");
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    ensure!(af4_usize(&record["bytes"])? == bytes.len() && record["sha256"].as_str() == Some(af4_digest(&bytes).as_str()), "{name}: actual stored buffer differs");
    Ok(())
}

fn af4_check_weights(manifest: &serde_json::Value, runtime: &DFlash2Runtime,
    adapter: &QwenDFlash2Adapter<'_>, model: &Qwen35Model, gguf: &mlx_native::gguf::GgufFile) -> Result<()> {
    use super::super::adapters::DFlash2TargetFamily;
    let inventory = manifest["weights"]["weights"].as_object().context("complete actual native weights missing")?;
    let c = &runtime.config;
    let t = &runtime.tensors;
    af4_check_weight_images(manifest, c.num_hidden_layers)?;
    let mut covered = std::collections::BTreeSet::<String>::new();
    let mut projection = |name: &str, p: &super::super::tensors::DFlash2Projection, n, k, first, count| -> Result<()> {
        af4_check_projection(inventory.get(name).with_context(|| format!("missing actual weight {name}"))?, p, n, k, first, count, name)?;
        covered.insert(name.to_string()); Ok(())
    };
    projection("context_projection", &t.context_projection, c.hidden_size, c.fc_input_dim(), 0, c.hidden_size)?;
    projection("selector_projection", &t.selector_projection, c.selector_rank, c.hidden_size, 0, c.selector_rank)?;
    let q = c.num_attention_heads * c.head_dim;
    let kv = c.num_key_value_heads * c.head_dim;
    for (index, layer) in t.layers.iter().enumerate() {
        let prefix = format!("layer.{index}");
        for (suffix, p, n, k, first, count) in [
            ("attention_dynamic", &layer.attention_dynamic, c.dynamic_projection_width(), c.hidden_size, 0, c.dynamic_projection_width()),
            ("qkv_projection", &layer.qkv_projection, q + 2 * kv, c.hidden_size, 0, q + 2 * kv),
            ("output_projection", &layer.output_projection, c.hidden_size, q, 0, c.hidden_size),
            ("mlp_dynamic", &layer.mlp_dynamic, c.dynamic_projection_width(), c.hidden_size, 0, c.dynamic_projection_width()),
            ("gate_projection", &layer.gate_up_projection, 2 * c.intermediate_size, c.hidden_size, 0, c.intermediate_size),
            ("up_projection", &layer.gate_up_projection, 2 * c.intermediate_size, c.hidden_size, c.intermediate_size, c.intermediate_size),
            ("down_projection", &layer.down_projection, c.hidden_size, c.intermediate_size, 0, c.hidden_size),
        ] { projection(&format!("{prefix}.{suffix}"), p, n, k, first, count)?; }
        ensure!(af4_logical_hashes(&layer.context_kv_projection, 2 * kv, c.hidden_size, 0, 2 * kv)?
            == af4_logical_hashes(&layer.qkv_projection, q + 2 * kv, c.hidden_size, q, 2 * kv)?, "{prefix}: context KV view differs");
        if let Some(halves) = &layer.gate_up_halves {
            for (half, first) in halves.iter().zip([0, c.intermediate_size]) {
                ensure!(af4_logical_hashes(half, c.intermediate_size, c.hidden_size, 0, c.intermediate_size)?
                    == af4_logical_hashes(&layer.gate_up_projection, 2 * c.intermediate_size, c.hidden_size, first, c.intermediate_size)?, "{prefix}: active gate/up view differs");
            }
        }
    }
    drop(projection);
    let mut buffer = |name: &str, value: &MlxBuffer, norm| -> Result<()> {
        af4_check_buffer(inventory.get(name).with_context(|| format!("missing stored weight {name}"))?, value, norm, name)?;
        covered.insert(name.to_string()); Ok(())
    };
    for (name, value, norm) in [("hidden_norm", &t.hidden_norm, true), ("final_norm", &t.final_norm, true),
        ("predecessor_codebook", &t.predecessor_codebook, false), ("successor_codebook", &t.successor_codebook, false)] {
        buffer(name, value, norm)?;
    }
    for (index, layer) in t.layers.iter().enumerate() {
        for (suffix, value, norm) in [("input_norm", &layer.input_norm, true), ("query_norm", &layer.query_norm, true),
            ("key_norm", &layer.key_norm, true), ("post_attention_norm", &layer.post_attention_norm, true),
            ("attention_convolution", &layer.attention_convolution, false), ("mlp_convolution", &layer.mlp_convolution, false)] {
            buffer(&format!("layer.{index}.{suffix}"), value, norm)?;
        }
    }
    let mask = "mask_embedding_not_applied_to_frozen_noise";
    if let Some(value) = &t.mask_embedding { buffer(mask, value, false)?; }
    drop(buffer);
    if t.mask_embedding.is_none() {
        ensure!(af4_usize(&inventory[mask]["bytes"])? == 0, "native mask/HF mask presence differs");
        covered.insert(mask.to_string());
    }
    let norm_bytes = model.output_norm.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    ensure!(model.output_norm.iter().all(|x| x.is_finite()) && inventory["head_final_norm_loaded_but_not_applied"]["sha256"] == af4_digest(&norm_bytes)
        && af4_usize(&inventory["head_final_norm_loaded_but_not_applied"]["bytes"])? == norm_bytes.len(), "actual target head norm identity differs");
    covered.insert("head_final_norm_loaded_but_not_applied".to_string());
    let head = adapter.vocabulary_head()?;
    ensure!(head.info.rows == c.vocab_size && head.info.cols == c.hidden_size && head.info.ggml_dtype == mlx_native::GgmlType::Q6_K
        && head.affine.is_none() && head.f16_shadow.is_none(), "actual target head geometry/storage differs");
    let name = if model.tied_word_embeddings { "token_embd.weight" } else { "output.weight" };
    let raw = gguf.read_tensor_bytes_host(name)?;
    ensure!(head.buffer.as_slice::<u8>()? == raw.as_slice(), "loaded actual target head differs from bound GGUF bytes");
    let native_head = &inventory["actual_target_vocabulary_head"];
    ensure!(native_head["destination"] == "F32" && af4_usize(&native_head["input_size"])? == c.hidden_size
        && af4_usize(&native_head["output_size"])? == c.vocab_size, "native actual head identity differs");
    let segments = native_head["segments"].as_array().context("actual head segments missing")?;
    ensure!(segments.len() == 1 && segments[0]["format"] == "q6k" && segments[0]["format_id"] == 4,
        "native actual head is not the bound Q6_K target head");
    for key in ["plane0", "plane1", "metadata"] {
        ensure!(af4_usize(&segments[0][key]["bytes"])? > 0 && segments[0][key]["sha256"].as_str().is_some_and(|s| s.len() == 64), "actual head physical inventory missing");
    }
    covered.insert("actual_target_vocabulary_head".to_string());
    ensure!(covered == inventory.keys().cloned().collect(), "native weight inventory has missing/unconsumed entries");
    Ok(())
}

fn af4_check_weight_images(manifest: &serde_json::Value, layers: usize) -> Result<()> {
    let inventory = &manifest["weights"];
    let files = inventory["files"].as_array().context("actual source weight files missing")?;
    let images = inventory["prepared_images"].as_array().context("actual prepared weight images missing")?;
    let mut expected = (0..layers).map(|i| format!("draft/layer-{i}.bin")).collect::<std::collections::BTreeSet<_>>();
    expected.extend(["draft/model.bin".to_string(), "target/head.bin".to_string()]);
    ensure!(files.len() == layers + 2 && images.len() == layers + 2, "complete source/prepared image count differs");
    let b = &manifest["bindings"];
    let identity = format!("{}:{}:{}", b["checkpoint_config_sha256"].as_str().context("config identity missing")?,
        b["checkpoint_weights_sha256"].as_str().context("weight identity missing")?,
        b["target_gguf_sha256"].as_str().context("target identity missing")?);
    let mut seen = std::collections::BTreeSet::new();
    for file in files {
        let path = file["relative_path"].as_str().context("prepared source path missing")?;
        ensure!(expected.contains(path) && seen.insert(path.to_string()) && file["magic"] == "MDGG0001"
            && file["content_identity"] == identity && af4_usize(&file["declared_bytes"])? > 0, "source weight file identity differs");
        let image = images.iter().find(|x| x["component"] == path).context("prepared weight image missing")?;
        ensure!(image["bytes"] == file["declared_bytes"]
            && image["sha256"].as_str().is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())),
            "prepared weight image digest/extent missing");
    }
    ensure!(seen == expected, "prepared image coverage differs");
    Ok(())
}
