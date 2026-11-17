// HF deliberately declares an uncompressed captured-hidden rebuild route, not peer lossy storage.
// Its own restore/crop/continuation byte invariants remain exact and independent of peer error.
fn af4_ring_bits(runtime: &DFlash2Runtime) -> Result<Vec<Vec<u16>>> {
    runtime.workspace.ring_keys.iter().chain(&runtime.workspace.ring_values)
        .map(|buffer| Ok(buffer.as_slice::<u16>()?.to_vec())).collect()
}

fn af4_ingest(runtime: &mut DFlash2Runtime, device: &MlxDevice, registry: &mut KernelRegistry,
    fixture: &SafeTensors<'_>, name: &str, rows: usize, start: usize) -> Result<()> {
    let c = &runtime.config;
    let lanes = runtime.workspace.max_lanes;
    let target = float_values(af4_tensor(fixture, name, Dtype::F32, &[lanes, rows, c.fc_input_dim()])?);
    let hidden = c.fc_input_dim() / c.target_layer_ids.len();
    let layer_ids = c.target_layer_ids.clone();
    for lane in 0..lanes {
        let capture = capture_lane(&target, lane, rows, hidden, &layer_ids);
        runtime.ingest_target_capture(device, registry, &capture, lane, start)?;
        runtime.record_lane_tokens(lane, &(start..start + rows).map(|v| v as u32).collect::<Vec<_>>())?;
    }
    Ok(())
}

fn af4_initialize_context(runtime: &mut DFlash2Runtime, device: &MlxDevice, registry: &mut KernelRegistry,
    fixture: &SafeTensors<'_>, rows: usize, start: usize) -> Result<()> {
    if start != 0 {
        ensure!(start == 12 && rows == 3, "restore original geometry differs");
        af4_ingest(runtime, device, registry, fixture, "input.initial_target_hidden", 12, 0)?;
        let before = af4_ring_bits(runtime)?;
        for lane in 0..runtime.workspace.max_lanes { runtime.clear_lane(lane)?; }
        ensure!(runtime.lane_lengths().iter().all(|&v| v == 0)
            && af4_ring_bits(runtime)?.iter().flatten().all(|&v| v == 0), "clear did not reset restore destination");
        // Re-execute context projection, normalization, KV and RoPE into empty rings.
        // The saved live KV above is only compared; it is NEVER copied into the runtime.
        af4_ingest(runtime, device, registry, fixture, "input.initial_target_hidden", 12, 0)?;
        ensure!(af4_ring_bits(runtime)? == before && runtime.lane_lengths().iter().all(|&v| v == 12),
            "HF uncompressed restore changed its own context bytes/cursor");
    }
    af4_ingest(runtime, device, registry, fixture, "input.target_hidden", rows, start)?;
    ensure!(runtime.lane_lengths().iter().all(|&v| v == start + rows), "context cursor differs after append");
    Ok(())
}

fn af4_validate_native_state(manifest: &serde_json::Value, fixture: &SafeTensors<'_>, runtime: &DFlash2Runtime) -> Result<()> {
    let c = &runtime.config;
    let lanes = runtime.workspace.max_lanes;
    let state = &manifest["state"];
    let position = af4_usize(&manifest["geometry"]["context_start"])? + af4_usize(&manifest["geometry"]["context_rows"])?;
    let valid = position.min(c.sliding_window);
    ensure!(af4_usize(&state["absolute_position"])? == position && af4_usize(&state["cursor"])? == position % c.sliding_window
        && af4_usize(&state["valid_rows"])? == valid, "native state position/cursor/validity differs");
    ensure!(state["canonical_layout"] == "lane,head,row,dim" && state["native_value_layout"] == "head,dim,row"
        && state["valid_ring_intervals"] == serde_json::json!([[0, valid]]), "native state layout/valid intervals differ");
    let positions = state["row_absolute_positions"].as_array().context("native row absolute positions missing")?;
    ensure!(positions.len() == c.sliding_window, "native state row position coverage differs");
    for (row, absolute) in positions.iter().enumerate() {
        if row < valid { ensure!(absolute.as_u64() == Some(row as u64), "native ring row position differs"); }
        else { ensure!(absolute.is_null(), "native invalid ring row declared valid"); }
    }
    let restoring = manifest["case"] == "restore";
    af4_context_coverage(manifest, fixture, c.hidden_size, lanes, restoring)?;
    ensure!(state["restored"] == restoring && af4_usize(&state["restore_initial_rows"])? == if restoring { 12 } else { 0 }, "native actual reconstruction metadata differs");
    let packed = state["packed_window_tensors"].as_array().context("packed native windows missing")?;
    let saved = state["saved_window_tensors"].as_array().context("saved native windows missing")?;
    ensure!(packed.len() == lanes && saved.is_empty(), "native diagnostic window/restore-input coverage differs");
    let packed_bytes = c.sliding_window * (c.hidden_size / 2 + c.hidden_size / 64 * 4);
    for lane in 0..lanes {
        let name = format!("native.state.lane{lane}.window");
        ensure!(packed[lane] == name, "native packed window lane differs");
        af4_tensor(fixture, &name, Dtype::U8, &[packed_bytes])?;
    }
    for layer in 0..c.num_hidden_layers {
        for kind in ["keys", "values"] {
            let name = format!("state.layer.{layer}.{kind}");
            let view = af4_tensor(fixture, &name, Dtype::BF16, &[lanes, c.num_key_value_heads, c.sliding_window, c.head_dim])?;
            let canonical = view.data();
            let lane_bytes = c.num_key_value_heads * c.sliding_window * c.head_dim * 2;
            for lane in 0..lanes {
                let raw = fixture.tensor(&format!("native.state.lane{lane}.layer{layer}.{kind}"))?;
                let shape = if kind == "keys" { [c.num_key_value_heads, c.sliding_window, c.head_dim] }
                    else { [c.num_key_value_heads, c.head_dim, c.sliding_window] };
                ensure!(raw.shape() == shape, "native raw state axes differ");
                ensure!(raw.dtype() == Dtype::BF16 && raw.data().len() == lane_bytes, "native raw state coverage differs");
                for head in 0..c.num_key_value_heads {
                    for row in 0..c.sliding_window {
                        for dim in 0..c.head_dim {
                            let at = ((head * c.sliding_window + row) * c.head_dim + dim) * 2;
                            let source = if kind == "keys" { at } else { ((head * c.head_dim + dim) * c.sliding_window + row) * 2 };
                            ensure!(canonical[lane * lane_bytes + at..lane * lane_bytes + at + 2] == raw.data()[source..source + 2], "{name}: canonical state is not the actual native byte view");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn af4_compare_state(fixture: &SafeTensors<'_>, runtime: &DFlash2Runtime) -> Result<()> {
    let c = &runtime.config;
    for layer in 0..c.num_hidden_layers {
        for (kind, buffer) in [("keys", &runtime.workspace.ring_keys[layer]), ("values", &runtime.workspace.ring_values[layer])] {
            let expected = float_values(fixture.tensor(&format!("state.layer.{layer}.{kind}"))?);
            let actual = buffer.as_slice::<u16>()?;
            ensure!(actual.len() == expected.len(), "semantic KV extent differs");
            let mut delta = 0.0_f32;
            let mut scale = 1.0_f32;
            for lane in 0..runtime.workspace.max_lanes {
                let position = runtime.lane_lengths()[lane];
                for head in 0..c.num_key_value_heads {
                    for absolute in position.saturating_sub(c.sliding_window)..position {
                        let row = absolute % c.sliding_window;
                        let at = ((lane * c.num_key_value_heads + head) * c.sliding_window + row) * c.head_dim;
                        for dim in 0..c.head_dim {
                            let a = bf16::from_bits(actual[at + dim]).to_f32();
                            let e = expected[at + dim];
                            ensure!(a.is_finite() && e.is_finite(), "nonfinite valid semantic KV row");
                            delta = delta.max((a - e).abs()); scale = scale.max(e.abs());
                        }
                    }
                }
            }
            eprintln!("NATIVE_AF4_STATE layer={layer} {kind} relative_max_abs={}", delta / scale);
            ensure!(delta / scale <= 0.05, "layer {layer} {kind}: valid semantic KV relative max_abs={} exceeds 0.05", delta / scale);
        }
    }
    Ok(())
}

fn af4_crop_continuation(runtime: &mut DFlash2Runtime, device: &MlxDevice, registry: &mut KernelRegistry,
    fixture: &SafeTensors<'_>, rows: usize, start: usize) -> Result<()> {
    let original = af4_ring_bits(runtime)?;
    let lengths = runtime.lane_lengths().to_vec();
    let c = &runtime.config;
    let values = float_values(fixture.tensor("input.target_hidden")?);
    let layer_ids = c.target_layer_ids.clone();
    let hidden = c.fc_input_dim() / layer_ids.len();
    // Crop only unwrapped prefixes. At/full past the wrap a rewind must reject byte-exactly.
    for lane in 0..runtime.workspace.max_lanes {
        let length = lengths[lane];
        let prefix = (0..length - 1).map(|v| v as u32).collect::<Vec<_>>();
        if length > runtime.config.sliding_window {
            ensure!(!runtime.align_lane_prefix(lane, &prefix), "wrapped crop unexpectedly succeeded");
        } else {
            ensure!(runtime.align_lane_prefix(lane, &prefix), "unwrapped crop failed");
            ensure!(runtime.lane_lengths()[lane] == length - 1 && af4_ring_bits(runtime)? == original,
                "crop changed physical KV bytes or wrong cursor");
            let capture = capture_lane(&values, lane, rows, hidden, &layer_ids);
            runtime.ingest_target_capture_rows(device, registry, &capture, lane, start + rows - 1, rows - 1..rows)?;
            runtime.record_lane_tokens(lane, &[(length - 1) as u32])?;
        }
        ensure!(runtime.lane_lengths() == lengths.as_slice() && af4_ring_bits(runtime)? == original,
            "crop/continuation changed own-route context bytes/cursors");
    }
    // Rejected context transaction must not modify any state or cursor.
    let capture = capture_lane(&values, 0, rows, hidden, &layer_ids);
    ensure!(runtime.ingest_target_capture(device, registry, &capture, 0, lengths[0] + 1).is_err(), "mispositioned context transaction accepted");
    ensure!(runtime.lane_lengths() == lengths.as_slice() && af4_ring_bits(runtime)? == original,
        "rejected context transaction modified bytes/cursors");
    Ok(())
}

fn af4_context_coverage(manifest: &serde_json::Value, fixture: &SafeTensors<'_>,
    hidden: usize, lanes: usize, restoring: bool) -> Result<()> {
    let rows = af4_usize(&manifest["geometry"]["context_rows"])?;
    for (input, count) in [("input.target_hidden", rows),
        ("input.initial_target_hidden", if restoring { 12 } else { 0 }),
        ("restore.input.initial_target_hidden", if restoring { 12 } else { 0 })] {
        if count == 0 { continue; }
        for lane in 0..lanes {
            let prefix = format!("native.prefill.{input}.lane{lane}.chunk");
            let mut chunks = Vec::new();
            for name in fixture.names() {
                if let Some(rest) = name.strip_prefix(&prefix).and_then(|s| s.strip_suffix(".native.context.projected")) {
                    chunks.push((rest.parse::<usize>()?, name));
                }
            }
            chunks.sort_unstable_by_key(|(offset, _)| *offset);
            ensure!(!chunks.is_empty(), "actual native context projection coverage missing");
            let mut cursor = 0;
            for (offset, name) in chunks {
                let projected = fixture.tensor(name)?;
                ensure!(offset == cursor && projected.shape().len() == 2 && projected.shape()[0] > 0
                    && projected.shape()[1] == hidden, "native context projection coverage is gapped");
                af4_tensor(fixture, name, Dtype::BF16, projected.shape())?;
                let other = name.strip_suffix("projected").context("native context projection suffix differs")?.to_string() + "hidden";
                af4_tensor(fixture, &other, Dtype::BF16, projected.shape())?;
                cursor += projected.shape()[0];
            }
            ensure!(cursor == count, "native original context row coverage differs");
        }
    }
    Ok(())
}
