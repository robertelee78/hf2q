//! Two root-run processes compare cold versus real cancellation/recovery tails.
//! All80 rows64..143 are captured; no full144-logit or whole-Q1 claim.
use super::*;
use crate::adr063_invariance::{
    acquisition, compare,
    protocol::Workload,
    qwen_cache, qwen_continuation as tail,
    receipt::*,
    store::{hash, Store},
};
use crate::inference::models::qwen35::spec_decode::positions_for_range;
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
};
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
enum Mode {
    ColdBaseline,
    CancelledThenRecovered,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Call {
    lane: usize,
    before: usize,
    input_rows: usize,
    after: usize,
    output_rows: usize,
}
fn require_calls(mode: Mode, calls: &[Call]) -> Result<()> {
    let mut expected = Vec::new();
    for lane in 0..4 {
        let (before, input_rows) = if mode == Mode::ColdBaseline {
            (0, 128)
        } else {
            (64, 64)
        };
        expected.push(Call {
            lane,
            before,
            input_rows,
            after: 128,
            output_rows: input_rows,
        });
        for before in 128..144 {
            expected.push(Call {
                lane,
                before,
                input_rows: 1,
                after: before + 1,
                output_rows: 1,
            });
        }
    }
    anyhow::ensure!(
        calls == expected,
        "actual continuation call/cursor/order differs from declared mode"
    );
    Ok(())
}
fn reset_and_check(
    model: &mut super::super::super::engine_qwen35::Qwen35LoadedModel,
) -> Result<usize> {
    let kv = model
        .persistent_kv_cache
        .as_mut()
        .context("persistent cache missing")?;
    kv.reset_for_slot(SlotId(0))?;
    anyhow::ensure!(
        kv.n_seqs == 1 && kv.sequence_len_for_slot(SlotId(0))? == 0,
        "fresh independent physical slot required"
    );
    let mut elements = 0usize;
    for state in &kv.linear_attn {
        for buffer in [
            &state.conv_state,
            &state.conv_state_scratch,
            &state.recurrent,
            &state.recurrent_scratch,
        ] {
            anyhow::ensure!(
                buffer.dtype() == mlx_native::DType::F32,
                "semantic-zero state scalar"
            );
            let values = buffer.as_logical_slice::<f32>()?;
            anyhow::ensure!(
                values.iter().all(|&v| v == 0.0),
                "source reset left nonzero recurrent/conv state"
            );
            elements = elements
                .checked_add(values.len())
                .context("zero-state count overflow")?;
        }
    }
    Ok(elements)
}
fn target_tail(
    model: &mut super::super::super::engine_qwen35::Qwen35LoadedModel,
    mode: Mode,
    lane: usize,
    tokens: &[u32],
    store: &mut Store,
    calls: &mut Vec<Call>,
) -> Result<()> {
    anyhow::ensure!(tokens.len() == 144, "fixed input144");
    let mut guard = Qwen35KvGuard::take(model)?;
    let start = if mode == Mode::ColdBaseline { 0 } else { 64 };
    let kv = guard.kv.as_ref().unwrap();
    anyhow::ensure!(
        kv.sequence_len_for_slot(SlotId(0))? as usize == start,
        "continuation must start from actual recovered or fresh cursor"
    );
    for (begin, end) in
        std::iter::once((start, 128)).chain((128..144).map(|position| (position, position + 1)))
    {
        let cache = guard.kv.as_mut().unwrap();
        let before = cache.sequence_len_for_slot(SlotId(0))? as usize;
        anyhow::ensure!(before == begin, "target begins at wrong actual cursor");
        let positions = positions_for_range(i32::try_from(begin)?, end - begin);
        let logits =
            guard
                .model
                .model
                .forward_gpu(&tokens[begin..end], &positions, cache, SlotId(0))?;
        let after = cache.sequence_len_for_slot(SlotId(0))? as usize;
        anyhow::ensure!(
            after == end
                && logits.len() == (end - begin) * guard.model.model.cfg.vocab_size as usize,
            "actual All target rows/cursor missing"
        );
        calls.push(Call {
            lane,
            before,
            input_rows: end - begin,
            after,
            output_rows: logits.len() / guard.model.model.cfg.vocab_size as usize,
        });
        tail::after_checked_target(store, &guard.model.model.cfg, lane, begin, &logits, cache)?;
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    acquisition: acquisition::Binding,
    model: PathBuf,
    output: PathBuf,
    identity: Identity,
    workload: Workload,
    mode: Mode,
    configured_context: u32,
    snapshot_payload_limit: u64,
    continuation_payload_limit: u64,
}
fn recipe<'a>(
    family: Family,
    capacity: u32,
    config_sha: String,
    cache_mode: &'a str,
) -> acquisition::ConfigRecipe<'a> {
    acquisition::ConfigRecipe {
        family,
        capacity: capacity as u64,
        slots: 1,
        output_allowance: 8192,
        cache_mode,
        route: "qwen-recovery-continuation-v1;targetAll;rows64..143",
        actual_model_config_sha256: config_sha,
    }
}
#[test]
#[ignore = "root-supervised real Qwen ordinary-specOFF model; independent resource/identity packet required"]
fn q1_qwen_actual_recovery_continuation_trace() {
    run().unwrap();
}
fn require_output_preflight(output: &std::path::Path) -> Result<()> {
    anyhow::ensure!(output.is_absolute(), "absolute new output required");
    match std::fs::symlink_metadata(output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        _ => anyhow::bail!("output already exists or cannot be checked"),
    }
    let parent = output.parent().context("output parent missing")?;
    anyhow::ensure!(
        parent.canonicalize()? == parent,
        "canonical output parent required"
    );
    anyhow::ensure!(
        !output
            .ancestors()
            .any(|ancestor| ancestor.join(".git").exists()),
        "diagnostic output must be outside Git ancestors"
    );
    Ok(())
}
fn run() -> Result<()> {
    use super::super::super::qwen35_speculation::{
        QwenSpeculationController, QwenSpeculationPolicy,
    };
    let path = PathBuf::from(
        std::env::var_os("HF2Q_Q1_QWEN_CONTINUATION_PACKET")
            .context("explicit continuation packet")?,
    );
    let mut raw = Vec::new();
    std::fs::File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    anyhow::ensure!(raw.len() <= 1024 * 1024, "packet1MiB");
    let packet: Packet = serde_json::from_slice(&raw)?;
    anyhow::ensure!(packet.execution_approved, "root approval missing");
    packet.workload.validate().map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        matches!(packet.identity.family, Family::Qwen38 | Family::Qwen36)
            && packet.identity.family == packet.workload.family,
        "explicit Qwen family only"
    );
    anyhow::ensure!(
        packet.configured_context == 262144
            && packet.snapshot_payload_limit <= 1024 * 1024 * 1024
            && packet.continuation_payload_limit <= 2 * 1024 * 1024 * 1024,
        "prospective fixed context/payload bounds"
    );
    require_output_preflight(&packet.output)?;
    let scope = acquisition::Scope::begin(
        &packet.acquisition,
        &packet.identity,
        &packet.workload,
        &packet.model,
        &packet.output,
        &raw,
    )?;
    let header = acquisition::open_model(&packet.model)?;
    anyhow::ensure!(
        header.metadata_string("general.architecture")
            == Some(if packet.identity.family == Family::Qwen38 {
                "qwen35"
            } else {
                "qwen35moe"
            }),
        "exact architecture, no fallback"
    );
    let mut header_cfg = crate::inference::models::qwen35::Qwen35Config::from_gguf(&header)?;
    header_cfg.max_position_embeddings = packet.configured_context;
    let ordinary = QwenSpeculationPolicy::Off;
    let cache_mode = format!("QwenTQ8:{ordinary:?}");
    scope.check_config(
        &packet.identity,
        &recipe(
            packet.identity.family,
            packet.configured_context,
            hash(format!("{header_cfg:?}").as_bytes()),
            &cache_mode,
        ),
    )?;
    drop(header);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let options = LoadOptions {
        model_path: packet.model,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
    };
    let LoadedModel::Qwen35(mut model) =
        LoadedModel::load_with_context(&options, Some(packet.configured_context))?
    else {
        anyhow::bail!("Qwen-only loader");
    };
    // This is the reviewed ordinary diagnostic axis, never a serving default.
    model.speculation = QwenSpeculationController::new(ordinary);
    anyhow::ensure!(
        model.speculation.policy() == QwenSpeculationPolicy::Off
            && model.tq_kv_active
            && model.model.kv_graft.is_none(),
        "ordinaryOff/TQ8/ungrafted source contract"
    );
    anyhow::ensure!(
        model.model.cfg.vocab_size == packet.workload.vocab_size,
        "actual vocabulary"
    );
    scope.check_config(
        &packet.identity,
        &recipe(
            packet.identity.family,
            packet.configured_context,
            hash(format!("{:?}", model.model.cfg).as_bytes()),
            &format!("QwenTQ8:{:?}", model.speculation.policy()),
        ),
    )?;
    model.model.ensure_gpu_cache_primed()?;
    model.provision_multi_seq_kv_for_slot_aware(1)?;
    let observed = model.model.with_gpu_cache_mut(|device, _| {
        scope.observed(device, &model as *const _ as usize as u64, 3)
    })?;
    anyhow::ensure!(!packet.output.exists(), "new output only");
    std::fs::create_dir(&packet.output)?;
    std::fs::set_permissions(&packet.output, std::fs::Permissions::from_mode(0o700))?;
    let expected = tail::expected(&model.model.cfg, packet.configured_context)?;
    let snapshot = tail::checkpoint_expected(&model.model.cfg, packet.configured_context)?;
    let tail_bytes = tail::payload(&expected)?;
    let snapshot_bytes = tail::payload(&snapshot)?;
    let total = tail_bytes
        .checked_add(if packet.mode == Mode::CancelledThenRecovered {
            snapshot_bytes
                .checked_mul(2)
                .context("snapshot quota overflow")?
        } else {
            0
        })
        .context("total quota overflow")?;
    anyhow::ensure!(
        total <= 4 * 1024 * 1024 * 1024
            && tail_bytes <= packet.continuation_payload_limit
            && snapshot_bytes <= packet.snapshot_payload_limit,
        "no truncation on disk quota refusal"
    );
    let tail_root = packet.output.join("continuation");
    let mut store = Store::new(
        &tail_root,
        packet.identity.clone(),
        if packet.mode == Mode::ColdBaseline {
            "T0Tail64To143"
        } else {
            "T6Tail64To143"
        }
        .into(),
        expected.clone(),
        packet.continuation_payload_limit,
        expected.len(),
    )
    .map_err(anyhow::Error::msg)?;
    let mut bound = observed.clone();
    bound.output_root = tail_root.clone();
    store.bind_acquisition(bound).map_err(anyhow::Error::msg)?;
    let mut checkpoints = if packet.mode == Mode::CancelledThenRecovered {
        let make = |name: &str, label: &str| -> Result<Store> {
            let root = packet.output.join(name);
            let mut s = Store::new(
                &root,
                packet.identity.clone(),
                label.into(),
                snapshot.clone(),
                packet.snapshot_payload_limit,
                snapshot.len(),
            )
            .map_err(anyhow::Error::msg)?;
            let mut binding = observed.clone();
            binding.output_root = root;
            s.bind_acquisition(binding).map_err(anyhow::Error::msg)?;
            Ok(s)
        };
        Some((
            make("before", "T6Checkpoint64")?,
            make("recovered", "T6ActualEngineRecovered64")?,
        ))
    } else {
        None
    };
    let mut recovery = Vec::new();
    let mut calls = Vec::new();
    let mut zero_elements = Vec::new();
    let result = (|| -> Result<()> {
        for lane in 0..4 {
            zero_elements.push(reset_and_check(&mut model)?);
            if let Some((before, after)) = checkpoints.as_mut() {
                recovery.push(run_loaded(
                    &mut model,
                    &packet.workload.lanes[lane][..128],
                    SamplingParams {
                        max_tokens: 8192,
                        ..SamplingParams::default()
                    },
                    |phase, cache| {
                        let target = match phase {
                            Phase::Checkpoint64 => &mut *before,
                            Phase::Recovered64 => &mut *after,
                        };
                        qwen_cache::after_checked_terminal(
                            target,
                            cache,
                            model_layers(&header_cfg),
                            &format!("agent-{lane}"),
                            64,
                        )
                    },
                )?);
            }
            target_tail(
                &mut model,
                packet.mode,
                lane,
                &packet.workload.lanes[lane],
                &mut store,
                &mut calls,
            )?;
        }
        require_calls(packet.mode, &calls)?;
        Ok(())
    })();
    if let Err(error) = result {
        store.fail(format!("actual continuation incomplete: {error:#}"));
        let _ = store.finish();
        if let Some((mut a, mut b)) = checkpoints {
            a.fail("continuation failed");
            b.fail("continuation failed");
            let _ = a.finish();
            let _ = b.finish();
        }
        return Err(error);
    }
    scope.check_unchanged()?;
    let trace = store.finish().map_err(anyhow::Error::msg)?;
    anyhow::ensure!(trace.coverage.errors.is_empty(), "tail coverage incomplete");
    let recovery_proof = if let Some((a, b)) = checkpoints {
        let left = a.finish().map_err(anyhow::Error::msg)?;
        let right = b.finish().map_err(anyhow::Error::msg)?;
        let policies = left
            .coverage
            .expected
            .iter()
            .map(|e| (e.at.stage, NumericPolicy::Exact))
            .collect();
        let compared = compare::compare_recovery(
            &packet.output.join("before"),
            &left,
            &packet.output.join("recovered"),
            &right,
            &policies,
        );
        anyhow::ensure!(
            compared.disposition == Disposition::ExactEqual,
            "actual recovery bytes differ"
        );
        Some(compared)
    } else {
        None
    };
    let evidence = serde_json::json!({"schema":"hf2q.q1.qwen-recovery-continuation.v1","mode":packet.mode,"packet_sha256":hash(&raw),"ordinary_policy":format!("{:?}",model.speculation.policy()),"calls":calls,"recovery":recovery,"recovery_proof":recovery_proof,"zero_state_elements_checked_per_lane":zero_elements,"captured_rows_per_lane":80,"endpoint_scope":[64,144],"first64_logits_captured":false,"active_cache_boundaries":[128,144],"planned_tensor_payload":total,"trace_records":trace.coverage.captured.len(),"physical_target_input_rows_per_lane":if packet.mode==Mode::ColdBaseline{144}else{208},"cancelled_unpublished_rows_per_lane":if packet.mode==Mode::ColdBaseline{0}else{64},"full_144_token_trajectory_processed":true,"full_144_logit_coverage":false,"whole_q1_qualified":false,"task_quality_claim":false});
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&evidence)?)?;
    println!("Q1_QWEN_RECOVERY_CONTINUATION_COMPLETE mode={:?} lanes=4 rows_per_lane=80 first64_logits=false whole_q1=false",packet.mode);
    Ok(())
}
fn model_layers(cfg: &crate::inference::models::qwen35::Qwen35Config) -> u32 {
    cfg.num_hidden_layers
}
#[test]
fn q1_continuation_call_axis_cannot_relabel_cold_as_recovered() {
    let mut calls = Vec::new();
    for lane in 0..4 {
        calls.push(Call {
            lane,
            before: 64,
            input_rows: 64,
            after: 128,
            output_rows: 64,
        });
        for before in 128..144 {
            calls.push(Call {
                lane,
                before,
                input_rows: 1,
                after: before + 1,
                output_rows: 1,
            });
        }
    }
    assert!(require_calls(Mode::CancelledThenRecovered, &calls).is_ok());
    assert!(require_calls(Mode::ColdBaseline, &calls).is_err());
    let mut bad = calls;
    bad[0].output_rows = 1;
    assert!(require_calls(Mode::CancelledThenRecovered, &bad).is_err());
}

#[test]
fn q1_continuation_output_preflight_rejects_repo_and_existing_before_load() {
    let owned = tempfile::tempdir().unwrap();
    let root = owned.path().canonicalize().unwrap();
    let output = root.join("new-output");
    assert!(require_output_preflight(&output).is_ok());
    assert!(!output.exists(), "preflight creates nothing");
    assert!(require_output_preflight(std::path::Path::new("relative-output")).is_err());
    std::fs::write(&output, b"owned-existing").unwrap();
    assert!(require_output_preflight(&output).is_err());
    std::fs::remove_file(&output).unwrap();
    std::fs::write(root.join(".git"), b"owned-worktree-marker").unwrap();
    assert!(require_output_preflight(&output).is_err());
    assert!(!output.exists());
}
