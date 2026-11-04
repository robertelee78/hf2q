//! Root-run-only phases: one preparation, then two acquired frozen comparisons.
use super::*;
use crate::adr063_invariance::{
    acquisition::{self, Binding, FileBinding, Scope as AcquisitionScope},
    qwen_continuation::payload,
    store::hash,
};
use serde::Deserialize;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::Path,
};
// Exactly two acquisition::open_model calls in this direct-loader route:
// runner header below, then Qwen35LoadedModel::load_with_context. The latter
// passes its same verified &gguf to config and weight loading; no dispatcher open.
const DIRECT_QWEN_REQUIRED_MODEL_OPENS: u64 = 2;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prefixes {
    family: Family,
    vocab_size: u32,
    prompts: [Vec<u32>; 4],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    first_fork: Option<bool>,
    /// Required exactly when `first_fork` is `Some(true)`; refused otherwise.
    first_fork_inputs: Option<FirstForkInputs>,
    target_layers: Option<bool>,
    mode: Mode,
    acquisition: Binding,
    identity: Identity,
    input: FileBinding,
    model: PathBuf,
    output: PathBuf,
}
/// The reviewed frozen-input bindings of a first-fork packet: the retained
/// `preparation-only.json` and its independent review receipt.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FirstForkInputs {
    preparation: FileBinding,
    review: FileBinding,
}
/// Independent review receipt of one T5 frozen file, schema
/// `hf2q.q1.qwen-t5-frozen-review.v1`, closed keys:
/// `schema`, `verdict` ("Accepted" admits; any other string refuses),
/// `frozen_file_sha256` (sha256 of the reviewed `frozen-workload.json`),
/// `preparation_sha256` (sha256 of the reviewed `preparation-only.json`),
/// `reviewer` (non-empty reviewer id) and `notes` (free text).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenReview {
    schema: String,
    verdict: String,
    frozen_file_sha256: String,
    preparation_sha256: String,
    reviewer: String,
    #[allow(dead_code)]
    notes: String,
}
pub(super) const FROZEN_REVIEW_SCHEMA: &str = "hf2q.q1.qwen-t5-frozen-review.v1";
/// Retained Q38 first-fork inputs; still pinned by the Qwen3.8-only target
/// layer localizer, and the hosted control of the reviewed-input rule.
pub(super) const Q38_RETAINED_FULL144_DIGEST: &str =
    "aeaec06b8c893e4b2445f993a66a250ff3af0b595f54c749b1bd6790073eafbc";
pub(super) const Q38_RETAINED_FROZEN_SHA256: &str =
    "0a4af3ac5d6dacc843536004a5d92be1353e4dd4f0c45cca63d776a6cc92b82b";
/// First-fork reviewed-input rule; pure so hosted controls exercise it.
/// `inputs` = (preparation-only.json bytes, review receipt bytes), both
/// already hash-bound by the packet. The frozen input must be the file the
/// preparation wrote and the review accepted, and the identity digest the
/// full-144 digest the preparation computed after generation.
pub(super) fn check_first_fork_inputs(first_fork: bool, inputs: Option<(&[u8], &[u8])>,
    input_sha256: &str, token_vectors_sha256: &str) -> Result<()> {
    let Some((preparation, review)) = inputs else {
        anyhow::ensure!(!first_fork, "first-fork packet requires reviewed preparation/review bindings");
        return Ok(());
    };
    anyhow::ensure!(first_fork, "first_fork_inputs only on explicit first-fork packets");
    let p: serde_json::Value = serde_json::from_slice(preparation)?;
    anyhow::ensure!(p["frozen_file_sha256"] == input_sha256 &&
        p["full144_digest_computed_after_generation"] == token_vectors_sha256 &&
        p["comparable_acquisition"] == false,
        "preparation receipt does not bind this frozen input/full144 digest as non-comparable preparation");
    let r: FrozenReview = serde_json::from_slice(review)?;
    anyhow::ensure!(r.schema == FROZEN_REVIEW_SCHEMA && r.verdict == "Accepted" &&
        !r.reviewer.trim().is_empty() && r.frozen_file_sha256 == input_sha256 &&
        r.preparation_sha256 == hash(preparation),
        "independent review must accept exactly this frozen file and preparation");
    Ok(())
}
fn read_bound(bound: &FileBinding) -> Result<Vec<u8>> {
    anyhow::ensure!(
        bound.path.is_absolute() && bound.sha256.len() == 64,
        "bound input"
    );
    let m = std::fs::symlink_metadata(&bound.path)?;
    anyhow::ensure!(
        m.is_file() && !m.file_type().is_symlink() && m.len() <= 1024 * 1024,
        "bound input extent/type"
    );
    let mut b = Vec::new();
    File::open(&bound.path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut b)?;
    anyhow::ensure!(
        b.len() <= 1024 * 1024 && hash(&b) == bound.sha256,
        "bound input hash"
    );
    Ok(b)
}
fn output_preflight(path: &Path) -> Result<()> {
    anyhow::ensure!(
        path.is_absolute()
            && path.parent().context("output parent")?.canonicalize()? == path.parent().unwrap()
            && !path.ancestors().any(|p| p.join(".git").exists()),
        "new canonical private output outside Git"
    );
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => anyhow::bail!("output exists/uncheckable"),
    }
}
fn recipe<'a>(
    family: Family,
    debug: String,
    mode: Mode,
    cache_mode: &'a str,
) -> acquisition::ConfigRecipe<'a> {
    acquisition::ConfigRecipe {
        family,
        capacity: 262144,
        slots: 1,
        output_allowance: 8192,
        cache_mode,
        route: if mode == Mode::Preparation {
            "qwen-t5-prefix128-workload-preparation-v1"
        } else {
            "qwen-t5-explicit-MTP-K3-forced-proposal-v1"
        },
        actual_model_config_sha256: debug,
    }
}
fn write(root: &Path, name: &str, value: &serde_json::Value) -> Result<()> {
    let b = serde_json::to_vec_pretty(value)?;
    anyhow::ensure!(b.len() <= 1024 * 1024, "metadata output bound");
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(root.join(name))?
        .write_all(&b)?;
    Ok(())
}
#[test]
#[ignore = "root-supervised actual Qwen model; preparation is not comparable acquisition"]
fn q1_qwen_actual_t5_speculation_phase() {
    run().unwrap();
}
fn run() -> Result<()> { run_inner(false,false) }
pub(super) fn run_first_fork() -> Result<()> { run_inner(true,false) }
#[test]
#[ignore="explicit frozen first-fork target layer packet; actual model; root supervised"]
fn q1_qwen_t5_target_layer_localizer_actual() {run_inner(true,true).unwrap();}
fn run_inner(first_fork: bool,target_layers:bool) -> Result<()> {
    let mut raw = Vec::new();
    File::open(PathBuf::from(
        std::env::var_os("HF2Q_Q1_QWEN_T5_PACKET").context("explicit T5 packet")?,
    ))?
    .take(1024 * 1024 + 1)
    .read_to_end(&mut raw)?;
    anyhow::ensure!(raw.len() <= 1024 * 1024, "packet bound");
    let packet: Packet = serde_json::from_slice(&raw)?;
    anyhow::ensure!(packet.execution_approved, "root execution approval missing");
    anyhow::ensure!(packet.first_fork == if first_fork {Some(true)} else {None},
        "explicit first-fork packet/test pairing; no old-entrypoint relabeling");
    anyhow::ensure!(packet.target_layers==if target_layers {Some(true)}else{None},
        "explicit target-layer packet/test pairing; no old capture relabeling");
    anyhow::ensure!(packet.first_fork_inputs.is_some() == first_fork,
        "first_fork_inputs required exactly on explicit first-fork packets");
    if first_fork {
        anyhow::ensure!(packet.mode != Mode::Preparation &&
            matches!(packet.identity.family, Family::Qwen38 | Family::Qwen36),
            "first fork runs only on a reviewed generated Qwen full144 workload");
    }
    if target_layers {
        anyhow::ensure!(packet.identity.family == Family::Qwen38 &&
            packet.identity.token_vectors_sha256 == Q38_RETAINED_FULL144_DIGEST,
            "only retained generated Q38 full144 workload; no input reselection");
    }
    output_preflight(&packet.output)?;
    let input = read_bound(&packet.input)?;
    if target_layers { anyhow::ensure!(hash(&input) == Q38_RETAINED_FROZEN_SHA256,
        "exact retained successful preparation file"); }
    let first_fork_inputs = packet.first_fork_inputs.as_ref()
        .map(|i| Ok::<_, anyhow::Error>((read_bound(&i.preparation)?, read_bound(&i.review)?)))
        .transpose()?;
    check_first_fork_inputs(first_fork,
        first_fork_inputs.as_ref().map(|(p, r)| (p.as_slice(), r.as_slice())),
        &hash(&input), &packet.identity.token_vectors_sha256)?;
    let (prefixes, frozen, vocab) = if packet.mode == Mode::Preparation {
        let p: Prefixes = serde_json::from_slice(&input)?;
        anyhow::ensure!(p.family == packet.identity.family, "family mismatch");
        acquisition::preparation_prefix_digest(p.family, &p.prompts, p.vocab_size)?;
        (p.prompts.to_vec(), None, p.vocab_size)
    } else {
        let f: Frozen = serde_json::from_slice(&input)?;
        f.workload.validate().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            f.workload.family == packet.identity.family,
            "frozen family mismatch"
        );
        (
            f.workload.lanes.iter().map(|p| p[..128].to_vec()).collect(),
            Some(f.clone()),
            f.workload.vocab_size,
        )
    };
    let scope = if let Some(f) = &frozen {
        AcquisitionScope::begin(
            &packet.acquisition,
            &packet.identity,
            &f.workload,
            &packet.model,
            &packet.output,
            &raw,
        )?
    } else {
        AcquisitionScope::begin_preparation(
            &packet.acquisition,
            &packet.identity,
            &prefixes,
            vocab,
            &packet.model,
            &packet.output,
            &raw,
        )?
    };
    let header = acquisition::open_model(&packet.model)?;
    anyhow::ensure!(
        header.metadata_string("general.architecture")
            == Some(match packet.identity.family {
                Family::Qwen38 => "qwen35",
                Family::Qwen36 => "qwen35moe",
                _ => anyhow::bail!("Qwen family only"),
            }),
        "exact architecture"
    );
    let mut cfg = Qwen35Config::from_gguf(&header)?;
    cfg.max_position_embeddings = 262144;
    anyhow::ensure!(cfg.vocab_size == vocab, "actual vocabulary");
    let mode_label = "QwenTQ8:MTP-CoherentAuto";
    scope.check_config(
        &packet.identity,
        &recipe(
            packet.identity.family,
            hash(format!("{cfg:?}").as_bytes()),
            packet.mode,
            mode_label,
        ),
    )?;
    let (committed_bytes, raw_bytes, paired_bytes) = if first_fork {
        (payload(&super::first_fork::committed(&cfg)?)?, payload(&super::first_fork::raw(&cfg))?,
            super::first_fork::payload_plan(&cfg,false)?.checked_add(
                super::first_fork::payload_plan(&cfg,true)?).context("first-fork pair overflow")?)
    } else {plan::paired_payload(&cfg, 262144)?};
    if first_fork {anyhow::ensure!(paired_bytes<=1024*1024*1024,
        "source first-fork pair exceeds1GiB before model/device load");}
    if target_layers {anyhow::ensure!(paired_bytes.checked_add(2*crate::inference::models::qwen35::q1_target_layer_capture::ARM_BYTES)
        ==Some(834_079_552),"selected two-arm layer evidence bound before model/device load");}
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
    let mut model = Qwen35LoadedModel::load_with_context(&options, Some(262144))?;
    model.speculation = super::super::super::qwen35_speculation::QwenSpeculationController::new(
        super::super::super::qwen35_speculation::QwenSpeculationPolicy::Auto,
    );
    anyhow::ensure!(
        model.tq_kv_active && model.model.mtp.is_some() && model.model.kv_graft.is_none(),
        "actual Auto/MTP/TQ8/ungrafted required"
    );
    scope.check_config(
        &packet.identity,
        &recipe(
            packet.identity.family,
            hash(format!("{:?}", model.model.cfg).as_bytes()),
            packet.mode,
            mode_label,
        ),
    )?;
    model.model.ensure_gpu_cache_primed()?;
    model.provision_multi_seq_kv_for_slot_aware(1)?;
    let preparation_observed_model_opens = if packet.mode == Mode::Preparation {
        Some(scope.check_preparation_loaded(DIRECT_QWEN_REQUIRED_MODEL_OPENS)?)
    } else {
        None
    };
    let observed = if packet.mode == Mode::Preparation {
        None
    } else {
        Some(model.model.with_gpu_cache_mut(|device, _| {
            scope.observed(
                device,
                &model as *const _ as usize as u64,
                DIRECT_QWEN_REQUIRED_MODEL_OPENS,
            )
        })?)
    };
    if let Some(f) = &frozen {
        for lane in 0..4 {
            anyhow::ensure!(
                f.forced_bad_ids[lane]
                    == plan::bad_proposal(
                        f.workload.lanes[lane][131],
                        vocab,
                        &model.eos_token_ids
                    )?,
                "frozen bad proposal differs from deterministic rule"
            );
        }
    }
    output_preflight(&packet.output)?;
    std::fs::create_dir(&packet.output)?;
    std::fs::set_permissions(&packet.output, std::fs::Permissions::from_mode(0o700))?;
    let make = |name: &str, expected: Vec<ExpectedTensor>| -> Result<Store> {
        let root = packet.output.join(name);
        let bytes = payload(&expected)?;
        let count = expected.len();
        let mut s = Store::new(
            &root,
            packet.identity.clone(),
            format!("QwenT5:{:?}:{name}", packet.mode),
            expected,
            bytes,
            count,
        )
        .map_err(anyhow::Error::msg)?;
        let mut binding = observed
            .as_ref()
            .context("preparation cannot create model trace")?
            .clone();
        binding.output_root = root;
        s.bind_acquisition(binding).map_err(anyhow::Error::msg)?;
        Ok(s)
    };
    let committed = if packet.mode == Mode::Preparation {
        None
    } else {
        Some(make("committed", if first_fork {super::first_fork::committed(&cfg)?}
            else {plan::committed(&cfg, 262144)?})?)
    };
    let raw_store = if packet.mode == Mode::ForcedProposal {
        Some(make("raw-verifier", if first_fork {super::first_fork::raw(&cfg)}
            else {plan::raw(&cfg)})?)
    } else {
        None
    };
    let mut active = Active::new(
        packet.mode,
        cfg.clone(),
        prefixes.clone(),
        frozen.clone(),
        model.eos_token_ids.clone(),
        committed,
        raw_store,
    );
    if first_fork { active.first_fork=Some(super::first_fork::make(&cfg,
        packet.mode==Mode::ForcedProposal,&make)?); }
    let _capture = install(active)?;
    let target_layer_scope=if target_layers {
        let frozen=frozen.as_ref().context("no new layer-localizer preparation")?;
        Some(crate::inference::models::qwen35::q1_target_layer_capture::install(
            make("target-layer-localizer",crate::inference::models::qwen35::q1_target_layer_capture::expected())?,
            packet.mode==Mode::ForcedProposal,[frozen.workload.lanes[0][129],frozen.workload.lanes[0][130]])?)
    }else{None};
    let mut zero_elements = Vec::new();
    let result = (|| -> Result<()> {
        for lane in 0..(if first_fork {1} else {4}) {
            let count = if first_fork {
                super::trajectory::run_lane_until(&mut model,packet.mode,&prefixes[lane],lane,131)?
            } else {
                super::trajectory::run_lane(&mut model,packet.mode,&prefixes[lane],lane)?
            };
            zero_elements.push(count);
        }
        Ok(())
    })();
    let mut active = finish()?;
    scope.check_unchanged()?;
    let complete = if first_fork {super::first_fork::validate(&active)} else {validate_complete(&active)};
    let mut failed = result
        .as_ref()
        .err()
        .or_else(|| complete.as_ref().err())
        .map(|e| format!("{e:#}"));
    let target_layer_metadata=if let Some(scope)=target_layer_scope {
        match scope.finish(failed.as_deref()) {
            Ok(value)=>Some(value),
            Err(error)=>{failed.get_or_insert_with(||format!("target-layer trace: {error:#}"));None}
        }
    }else{None};
    if let Some(error) = &failed {
        if let Some(s) = active.committed.as_mut() {
            s.fail(error.clone());
        }
        if let Some(s) = active.raw.as_mut() {
            s.fail(error.clone());
        }
    }
    let first_fork_traces = active.first_fork.take().map(|c|
        super::first_fork::finish(c,failed.as_deref())).transpose()?;
    let committed_trace = active
        .committed
        .take()
        .map(Store::finish)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let raw_trace = active
        .raw
        .take()
        .map(Store::finish)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let mut descriptor = serde_json::json!({"schema":"hf2q.q1.qwen-t5-phase.v1","mode":packet.mode,"packet_sha256":hash(&raw),"input_sha256":hash(&input),"status":if failed.is_some(){"FailedOrIncomplete"}else if packet.mode==Mode::Preparation{"CompleteWorkloadPreparationNotComparable"}else{"CompleteScopedAcquisition"},"error":failed,"actual_transitions":active.transitions,"ordinary_input_positions":active.ordinary_rows,"canonical_decisions":active.selected,"semantic_zero_elements_checked":zero_elements,"committed_payload_plan":committed_bytes,"raw_payload_plan":raw_bytes,"paired_payload_plan":paired_bytes,"committed_records":committed_trace.as_ref().map(|t|t.coverage.captured.len()),"raw_records":raw_trace.as_ref().map(|t|t.coverage.captured.len()),"explicit_selector":"actual private MTP-K3 after real ordinary warmup, bypasses history preference only","forced_proposals":packet.mode==Mode::ForcedProposal,"ordinary_AUTO_selection_qualified":false,"proposal_quality_qualified":false,"speed_qualified":false,"whole_q1_qualified":false});
    if first_fork {
        descriptor["status"]=serde_json::json!(if failed.is_some() {"FailedOrIncomplete"}
            else {"CompleteFirstForkDiagnosticNotQ1Qualified"});
        descriptor["first_fork"]=serde_json::json!({"logical_lanes_executed":1,
            "logical_lane":0,"physical_slots":1,"committed_input_positions":[128,129,130],
            "cache_boundaries":[129,131],"capture_stores":first_fork_traces,
            "case_payload_plan":super::first_fork::payload_plan(&cfg,packet.mode==Mode::ForcedProposal)?,
            "initial129_exact_comparison_pending":true,"float_policy":"Unspecified",
            "all4lane_execution":false,"rejected_tail_baseline_claim":false,
            "reviewed_inputs":packet.first_fork_inputs.as_ref().map(|i|serde_json::json!({
                "preparation_sha256":i.preparation.sha256,"review_sha256":i.review.sha256,
                "review_schema":FROZEN_REVIEW_SCHEMA}))});
    }
    if target_layers {
        if let Some(metadata)=target_layer_metadata {
            write(&packet.output,"target-layer-localizer-metadata.json",&metadata)?;
        }
        descriptor["status"]=serde_json::json!(if failed.is_some() {"FailedOrIncomplete"}
            else {"CompleteTargetLayerDiagnosticNotQ1Qualified"});
        descriptor["target_layer_localizer"]=serde_json::json!({"records":258,"row_positions":[129,130],
            "payload_bytes":crate::inference::models::qwen35::q1_target_layer_capture::ARM_BYTES,
            "paired_total_payload_bytes":834_079_552,"additional_checked_waits":0,
            "native_operator_route":"UnknownNotObserved","whole_q1_qualified":false});
    }
    write(&packet.output, "execution.json", &descriptor)?;
    if let Some(error) = failed {
        anyhow::bail!(error);
    }
    if target_layers {
        println!("Q1_QWEN_T5_TARGET_LAYERS_COMPLETE mode={:?} logical_lanes=1 agent=0 cursor=131 checkpoints=258 float_policy=Unspecified whole_q1=false",packet.mode);
        return Ok(());
    }
    if first_fork {
        println!("Q1_QWEN_T5_FIRST_FORK_COMPLETE mode={:?} logical_lanes=1 agent=0 cursor=131 float_policy=Unspecified whole_q1=false",packet.mode);
        return Ok(());
    }
    if packet.mode == Mode::Preparation {
        let mut lanes = Vec::new();
        let mut carries = Vec::new();
        let mut bad = Vec::new();
        for lane in 0..4 {
            let mut tokens = active.prefixes[lane].clone();
            tokens.extend(active.selected[lane][..16].iter().map(|d| d.token));
            lanes.push(tokens);
            carries.push(active.selected[lane][16]);
            bad.push(plan::bad_proposal(
                active.selected[lane][3].token,
                vocab,
                &active.eos,
            )?);
        }
        let prepared = Frozen {
            workload: Workload {
                family: packet.identity.family,
                vocab_size: vocab,
                lanes,
            },
            final_carry: carries
                .try_into()
                .map_err(|_| anyhow::anyhow!("carry cardinality"))?,
            forced_bad_ids: bad
                .try_into()
                .map_err(|_| anyhow::anyhow!("bad ID cardinality"))?,
        };
        write(
            &packet.output,
            "frozen-workload.json",
            &serde_json::to_value(&prepared)?,
        )?;
        write(
            &packet.output,
            "preparation-only.json",
            &serde_json::json!({"source_commit":packet.identity.source_commit,"binary_sha256":packet.identity.binary_sha256,"prefix_digest":packet.identity.token_vectors_sha256,"required_model_opens":DIRECT_QWEN_REQUIRED_MODEL_OPENS,"observed_model_opens":preparation_observed_model_opens.context("preparation descriptor observation absent")?,"full144_digest_computed_after_generation":acquisition::tokens_digest(&prepared.workload)?,"frozen_file_sha256":hash(&std::fs::read(packet.output.join("frozen-workload.json"))?),"final_carry_is_metadata_not17th_input":true,"comparable_acquisition":false,"execution_approval_for_future_phases":false,"requires_independent_review":true}),
        )?;
    }
    println!("Q1_QWEN_T5_PHASE_COMPLETE mode={:?} logical_lanes=4 explicit_MTP_selector=true forced_proposal={} whole_q1=false",packet.mode,packet.mode==Mode::ForcedProposal);
    Ok(())
}
pub(super) fn validate_complete(active: &Active) -> Result<()> {
    anyhow::ensure!(
        active.selected.iter().all(|s| s.len() == 17),
        "seed+16actual canonical decisions/carry required"
    );
    for lane in 0..4 {
        let expected = if active.mode == Mode::ForcedProposal {
            std::iter::once(128).chain(131..144).collect::<Vec<_>>()
        } else {
            (128..144).collect()
        };
        anyhow::ensure!(
            active.ordinary_rows[lane] == expected,
            "ordinary transaction input positions omitted/reordered"
        );
    }
    anyhow::ensure!(
        active.pending_logits.is_none(),
        "verifier raw snapshot unconsumed"
    );
    anyhow::ensure!(
        active.transitions.len()
            == if active.mode == Mode::ForcedProposal {
                4
            } else {
                0
            },
        "actual K3 transition count incomplete"
    );
    Ok(())
}
