// Validate actual replay/restore bytes before loading a candidate model.
// This is a numerical reference policy, not a production compact-cache claim.
fn af4_require_reference_policy(manifest: &serde_json::Value) -> Result<()> {
    ensure!(manifest["reference_policy"] == serde_json::json!({
        "revision": AF4_REFERENCE_REVISION,
        "restore_input": "original-captured-target-hidden",
        "rebuild_api": "DFlashDraft::addContextPrefill",
    }), "native-AF4 matched reconstruction policy differs");
    Ok(())
}

fn af4_verify_lossless_restore(manifest: &serde_json::Value, fixture: &SafeTensors<'_>) -> Result<()> {
    af4_require_reference_policy(manifest)?;
    let restoring = manifest["case"] == "restore";
    let state = &manifest["state"];
    ensure!(state["restore_kind"] == if restoring { "captured-hidden-rebuild" } else { "none" },
        "native restore kind differs");
    ensure!(state["saved_window_tensors"] == serde_json::json!([]),
        "packed diagnostic window was labeled a lossless restore input");
    let replay_name = "native.restore.input.initial_target_hidden";
    let expected_inputs = if restoring { vec![replay_name] } else { Vec::new() };
    ensure!(state["restore_input_tensors"] == serde_json::to_value(expected_inputs)?,
        "native consumed restore input coverage differs");
    if !restoring {
        ensure!(!fixture.names().iter().any(|name| name.starts_with("native.restore.")),
            "non-restore execution contains reconstruction evidence");
        return Ok(());
    }
    let g = &manifest["geometry"];
    let lanes = af4_usize(&g["lanes"])?;
    let layers = af4_usize(&g["layers"])?;
    let heads = af4_usize(&g["kv_heads"])?;
    let window = af4_usize(&g["draft_window"])?;
    let dimension = af4_usize(&g["head_dimension"])?;
    let target = af4_usize(&g["target_hidden_size"])?;
    ensure!(lanes > 0 && layers > 0 && heads > 0 && dimension > 0 && window >= 12,
        "invalid lossless restore geometry");
    let initial = fixture.tensor("native.input.initial_target_hidden")?;
    let replay = fixture.tensor(replay_name)?;
    ensure!(initial.dtype() == Dtype::BF16 && initial.shape() == [lanes, 12, target]
        && replay.dtype() == initial.dtype() && replay.shape() == initial.shape()
        && replay.data() == initial.data(), "actual restore replay input differs from original rounded input");
    // The bundle validator independently binds initial to original F32 rounded
    // once, and checks every retained floating payload for finiteness.
    for lane in 0..lanes {
        for layer in 0..layers {
            for kind in ["keys", "values"] {
                let shape = if kind == "keys" { [heads, window, dimension] }
                    else { [heads, dimension, window] };
                let before = fixture.tensor(&format!("native.restore.lane{lane}.before.layer{layer}.{kind}"))?;
                let after = fixture.tensor(&format!("native.restore.lane{lane}.after.layer{layer}.{kind}"))?;
                let committed = fixture.tensor(&format!("native.state.lane{lane}.layer{layer}.{kind}"))?;
                for view in [&before, &after, &committed] {
                    ensure!(view.dtype() == Dtype::BF16 && view.shape() == shape,
                        "actual lossless restore ring dtype/shape differs");
                }
                ensure!(before.data() == after.data(), "native lossless rebuild changed complete ring bytes");
                for head in 0..heads {
                    for row in 0..window {
                        for dim in 0..dimension {
                            let at = if kind == "keys" { ((head * window + row) * dimension + dim) * 2 }
                                else { ((head * dimension + dim) * window + row) * 2 };
                            let rebuilt = &after.data()[at..at + 2];
                            if row < 12 {
                                ensure!(rebuilt == &committed.data()[at..at + 2],
                                    "rebuilt prefix changed in final native state");
                            } else {
                                ensure!(rebuilt == &[0, 0], "unused lossless restore ring row is not fresh zero storage");
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn native_af4_lossless_restore_rejects_incoherent_actual_bytes() -> Result<()> {
    let manifest = serde_json::json!({
        "case": "restore",
        "reference_policy": {
            "revision": AF4_REFERENCE_REVISION,
            "restore_input": "original-captured-target-hidden",
            "rebuild_api": "DFlashDraft::addContextPrefill"
        },
        "geometry": {"lanes": 1, "layers": 1, "kv_heads": 1, "draft_window": 16,
            "head_dimension": 2, "target_hidden_size": 2},
        "state": {"restore_kind": "captured-hidden-rebuild", "saved_window_tensors": [],
            "restore_input_tensors": ["native.restore.input.initial_target_hidden"]}
    });
    let mut tensors = std::collections::BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    for name in ["native.input.initial_target_hidden", "native.restore.input.initial_target_hidden"] {
        tensors.insert(name.into(), (vec![1, 12, 2], vec![0; 48]));
    }
    for kind in ["keys", "values"] {
        let shape = if kind == "keys" { vec![1, 16, 2] } else { vec![1, 2, 16] };
        for phase in ["before", "after"] {
            tensors.insert(format!("native.restore.lane0.{phase}.layer0.{kind}"), (shape.clone(), vec![0; 64]));
        }
        tensors.insert(format!("native.state.lane0.layer0.{kind}"), (shape, vec![0; 64]));
    }
    let encode = |tensors: &std::collections::BTreeMap<String, (Vec<usize>, Vec<u8>)>| -> Result<Vec<u8>> {
        let views = tensors.iter().map(|(name, (shape, data))| {
            Ok((name.as_str(), TensorView::new(Dtype::BF16, shape.clone(), data)?))
        }).collect::<Result<Vec<_>>>()?;
        Ok(safetensors::tensor::serialize(views, None)?)
    };
    let original = encode(&tensors)?;
    af4_verify_lossless_restore(&manifest, &SafeTensors::deserialize(&original)?)?;
    // Corrupt a real replay value, not its descriptor/affirmative witness.
    let mut bad = tensors.clone();
    bad.get_mut("native.restore.input.initial_target_hidden").unwrap().1[1] = 0x3f;
    let bytes = encode(&bad)?;
    assert!(af4_verify_lossless_restore(&manifest, &SafeTensors::deserialize(&bytes)?).is_err());
    // A changed live prefix cannot be called an exact reconstruction.
    let mut bad = tensors.clone();
    bad.get_mut("native.restore.lane0.before.layer0.keys").unwrap().1[1] = 0x3f;
    let bytes = encode(&bad)?;
    assert!(af4_verify_lossless_restore(&manifest, &SafeTensors::deserialize(&bytes)?).is_err());
    // Equal snapshots must not conceal stale unused rows (native V layout).
    let mut bad = tensors.clone();
    for phase in ["before", "after"] {
        bad.get_mut(&format!("native.restore.lane0.{phase}.layer0.values")).unwrap().1[29] = 0x3f;
    }
    let bytes = encode(&bad)?;
    assert!(af4_verify_lossless_restore(&manifest, &SafeTensors::deserialize(&bytes)?).is_err());
    // Valid rebuild snapshots cannot be paired with another policy's final KV.
    let mut bad = tensors.clone();
    bad.get_mut("native.state.lane0.layer0.keys").unwrap().1[1] = 0x3f;
    let bytes = encode(&bad)?;
    assert!(af4_verify_lossless_restore(&manifest, &SafeTensors::deserialize(&bytes)?).is_err());
    let mut bad = tensors;
    bad.remove("native.restore.lane0.after.layer0.values");
    let bytes = encode(&bad)?;
    assert!(af4_verify_lossless_restore(&manifest, &SafeTensors::deserialize(&bytes)?).is_err());
    let mut old_policy = manifest;
    old_policy["reference_policy"]["rebuild_api"] = "DFlashDraft::addContextRebuild".into();
    assert!(af4_verify_lossless_restore(&old_policy, &SafeTensors::deserialize(&original)?).is_err());
    Ok(())
}
