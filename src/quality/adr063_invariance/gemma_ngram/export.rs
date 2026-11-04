//! No-model Gemma T5 / localizer manifest export (supervisor manifest form):
//! `{lanes, host_payload_limit, expected_stores}` from the SAME source
//! manifest functions the adapter admits, read from the GGUF header only.
use super::{localizer,runner::t5_run::{Branch,Manifest,MANIFEST_BYTES},t5::{self,Arm}};
use crate::adr063_invariance::{acquisition,store::hash};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use serde_json::json;
use std::{fs::{File,OpenOptions},io::{Read,Write},os::unix::fs::{OpenOptionsExt,PermissionsExt},path::PathBuf};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {model:PathBuf,output:PathBuf,#[serde(default)] localizer:Option<Pass>}
/// One localizer pass. Pass 1 is all layers with the three onset roles;
/// pass 2 takes the per-lane pass-1 onsets and applies the frozen union rule.
#[derive(Clone,Debug,Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pass {
    pub(super) branch:Branch,pub(super) pass:u8,pub(super) rerun_prefix:String,
    #[serde(default)] pub(super) accepted:Option<[u8;4]>,
    #[serde(default)] pub(super) onset_layers:Option<[Option<u32>;4]>,
}
/// The pass Plan of one arm (the packet's verbatim `stage_localizer` block).
pub(super) fn plan(cfg:&crate::serve::config::Gemma4Config,l:&Pass,arm:Arm)->Result<localizer::Plan> {
    use crate::adr063_invariance::{gemma_cohort_localizer::DECODE_STAGES,gemma_localizer::PREFILL_STAGES};
    let (layers,prefill,decode)=match (l.pass,&l.onset_layers) {
        (1,None)=>((0..30).collect(),localizer::PASS1_STAGES.to_vec(),localizer::PASS1_STAGES.to_vec()),
        (2,Some(onsets))=>(localizer::pass2_layers(onsets).map_err(anyhow::Error::msg)?,PREFILL_STAGES.to_vec(),DECODE_STAGES.to_vec()),
        _=>anyhow::bail!("pass 1 takes no onsets; pass 2 requires the pass-1 per-lane onsets"),
    };
    ensure!(l.branch==Branch::N || (arm==Arm::OrdinaryGlobal && l.accepted.is_none()),"branch P: ordinary arm, no accepted count");
    ensure!(l.branch==Branch::P || l.accepted.is_some(),"branch N freezes the T5 n-gram accepted counts");
    let mut plan=localizer::Plan {rerun_id:format!("{}-{}-p{}",l.rerun_prefix,arm.label(),l.pass),layers,prefill_stages:prefill,
        decode_stages:decode,host_payload_limit:0,accepted:l.accepted};
    let (_,_,bytes)=localizer::manifest(cfg,&plan.layers,&plan.prefill_stages,&plan.decode_stages,arm,plan.accepted().map_err(anyhow::Error::msg)?)
        .map_err(anyhow::Error::msg)?;
    plan.host_payload_limit=bytes;localizer::require_pass(&plan,l.pass).map_err(anyhow::Error::msg)?;
    Ok(plan)
}
/// The manifest-form extent of one T5 arm (with the stage store when a Plan is given).
pub(super) fn manifest(cfg:&crate::serve::config::Gemma4Config,arm:Arm,plan:Option<&localizer::Plan>)->Result<Manifest> {
    let mut expected_stores=t5::stores(cfg,arm).map_err(anyhow::Error::msg)?.into_iter().map(|(n,e)|(n.to_string(),e)).collect::<std::collections::BTreeMap<_,_>>();
    let mut stage_limit=0;
    if let Some(plan)=plan {
        let (_,stages,_)=localizer::manifest(cfg,&plan.layers,&plan.prefill_stages,&plan.decode_stages,arm,plan.accepted().map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?;
        expected_stores.insert(t5::STAGES.into(),stages);stage_limit=plan.host_payload_limit;
    }
    Ok(Manifest {lanes:vec![0,1,2,3],host_payload_limit:t5::host_payload_limit(cfg,arm,stage_limit).map_err(anyhow::Error::msg)?,expected_stores})
}
/// `(file name, bytes)` of every exported file, `prepared.json` first.
pub(super) fn files(cfg:&crate::serve::config::Gemma4Config,l:Option<&Pass>)->Result<Vec<(String,Vec<u8>)>> {
    let cases:Vec<(String,Arm,Option<localizer::Plan>)>=match l {
        None=>vec![("t5ord".into(),Arm::OrdinaryGlobal,None),("t5ngram".into(),Arm::NgramTransaction,None)],
        Some(l)=>{
            let arms=match l.branch {Branch::N=>vec![Arm::OrdinaryGlobal,Arm::NgramTransaction],Branch::P=>vec![Arm::OrdinaryGlobal]};
            arms.into_iter().map(|arm|Ok((format!("t5loc-{}-p{}",if arm==Arm::OrdinaryGlobal {"ord"}else{"ngram"},l.pass),arm,Some(plan(cfg,l,arm)?))))
                .collect::<Result<_>>()?
        }
    };
    let mut files=Vec::new();let mut rows=Vec::new();
    for (case,arm,plan) in cases {
        let m=manifest(cfg,arm,plan.as_ref())?;
        let encoded=serde_json::to_vec_pretty(&m)?;ensure!(encoded.len() as u64<=MANIFEST_BYTES,"manifest file exceeds 64 MiB");
        let name=format!("manifest-{case}.json");
        rows.push(json!({"case":case,"manifest":{"file":name,"sha256":hash(&encoded)},
            "packet_extent":{"arm":arm,"lanes":m.lanes,"host_payload_limit":m.host_payload_limit,"stage_localizer":plan,
                "branch":l.map(|l|l.branch),"pass":l.map(|l|l.pass)},
            "store_records":m.expected_stores.iter().map(|(n,e)|(n.clone(),json!(e.len()))).collect::<serde_json::Map<_,_>>()}));
        files.push((name,encoded));
    }
    let prepared=json!({"schema":"hf2q.q1.gemma-t5-manifest-export.v1","native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "actual_gemma_config_debug_sha256":hash(format!("{cfg:?}").as_bytes()),
        "config_sha256":acquisition::config_digest(&super::configuration_recipe(cfg,true))?,"cases":rows,
        "routing_witness_charge":t5::witness_charge(),"packet_env":"HF2Q_Q1_GEMMA_T5_PACKET",
        "filters":{"t5":"adr063_invariance::gemma_ngram::runner::q1_gemma_actual_t5_trace",
            "localizer":"adr063_invariance::gemma_ngram::runner::q1_gemma_actual_t5_stage_localizer"},
        "markers":{"t5":t5::MARKER,"localizer":t5::LOCALIZER_MARKER},
        "model_loaded":false,"device_created":false,"inference_performed":false,"execution_approved":false,"whole_q1_qualified":false});
    files.insert(0,("prepared.json".into(),serde_json::to_vec_pretty(&prepared)?));
    Ok(files)
}
#[test]
#[ignore="root-supervised GGUF header only; no model/device"]
fn q1_gemma_t5_manifest_export(){run().unwrap();}
fn run()->Result<()> {
    let path=std::env::var_os("HF2Q_Q1_GEMMA_T5_MANIFEST_CONFIG").context("explicit Gemma T5 manifest config")?;
    let mut raw=Vec::new();File::open(path)?.take(65536+1).read_to_end(&mut raw)?;ensure!(raw.len()<=65536,"bounded export config");
    let c:Config=serde_json::from_slice(&raw)?;
    ensure!(c.output.is_absolute() && c.output.parent().context("output parent")?.canonicalize()?==c.output.parent().unwrap() &&
        !c.output.ancestors().any(|p|p.join(".git").exists()) &&
        matches!(std::fs::symlink_metadata(&c.output),Err(e) if e.kind()==std::io::ErrorKind::NotFound),"new private output outside Git");
    ensure!(c.model.is_absolute(),"absolute model path");
    let before=acquisition::FileIdentity::observed(&File::open(&c.model)?)?;
    let gguf=acquisition::open_model(&c.model)?;
    ensure!(gguf.tensor_data_offset()<=32*1024*1024 && gguf.metadata_string("general.architecture")==Some("gemma4"),"bounded Gemma header only");
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&gguf)?;cfg.max_position_embeddings=262144;
    ensure!(cfg.num_hidden_layers==30 && cfg.vocab_size==262144 && cfg.sliding_window>=148 && (0..30).all(|l|cfg.is_full_attention(l)==((l+1)%6==0)),
        "selected Gemma geometry");
    let files=files(&cfg,c.localizer.as_ref())?;
    ensure!(acquisition::FileIdentity::observed(&File::open(&c.model)?)?==before,"model changed during header read");
    std::fs::create_dir(&c.output)?;std::fs::set_permissions(&c.output,std::fs::Permissions::from_mode(0o700))?;
    for (name,bytes) in &files {OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join(name))?.write_all(bytes)?;}
    println!("Q1_GEMMA_T5_MANIFEST_EXPORTED files={} config_input_sha256={} model=false device=false execution=false",files.len(),hash(&raw));
    Ok(())
}
