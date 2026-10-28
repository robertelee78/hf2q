//! Root-only, one genuine transaction per serial owner; dataset binds later.
use super::*;
use crate::adr063_invariance::{acquisition::{self,FileBinding},protocol::Workload,receipt::{Family,Identity}};
use serde::Deserialize;
use std::{fs::File,io::Read,path::PathBuf};
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved:bool,identity:Identity,acquisition:acquisition::Binding,
    model:PathBuf,output:PathBuf,greedy_workload:FileBinding,independent_dataset_review:FileBinding,
    configured_context:u32,configured_output_allowance:u32,logical_serial_owners:usize,
    proposal_origin:String,ngram_k:u32,ngram_min:u32,ngram_max:u32,payload_limit:u64,metadata_limit:u64,
    #[serde(default)] routing_witness:Option<crate::adr063_invariance::gemma_routing_witness::Budget>,
}
fn preflight(p:&Packet)->Result<()> {
    p.identity.validate().map_err(anyhow::Error::msg)?;selected_profile()?;
    ensure!(p.execution_approved && p.identity.family==Family::Gemma4 &&
        p.identity.model_sha256=="bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53","approved selected Gemma only");
    ensure!(p.configured_context==262144 && p.configured_output_allowance==8192 && p.logical_serial_owners==4 &&
        p.ngram_k==3 && p.ngram_min==1 && p.ngram_max==3 && p.proposal_origin=="InjectedDiagnosticFixtureNotProductionProposalQuality" &&
        p.payload_limit==PAYLOAD && p.metadata_limit==METADATA,"closed finite selected transaction profile");
    ensure!(p.output.is_absolute() && p.output.parent().context("output parent")?.canonicalize()?==p.output.parent().unwrap() &&
        !p.output.ancestors().any(|p|p.join(".git").exists()) &&
        matches!(std::fs::symlink_metadata(&p.output),Err(e) if e.kind()==std::io::ErrorKind::NotFound),"new private output before load");
    for flag in ["HF2Q_DFLASH_XLEN_SDPA","HF2Q_DFLASH_BATCH_ARGMAX","HF2Q_SPEC_DFLASH","HF2Q_SPEC_EAGLE3"] {
        ensure!(std::env::var(flag).ok().is_none_or(|v|v=="0"),"unreviewed route override {flag}");
    }crate::adr063_invariance::gemma_routing_witness::validate_budget(p.routing_witness.as_ref())?;Ok(())
}
fn recipe(cfg:&crate::serve::config::Gemma4Config)->acquisition::ConfigRecipe<'static> {
    acquisition::ConfigRecipe {family:Family::Gemma4,capacity:262144,slots:4,output_allowance:8192,
        cache_mode:"GemmaHybrid-global-OptionC-serial-owner-epochs-no-xlen",
        route:"gemma-ngram-injected-one-real-round-then-ordinary-v1",
        actual_model_config_sha256:crate::adr063_invariance::store::hash(format!("{cfg:?}").as_bytes())}
}
/// Narrow parent-only accessor; metadata export and runtime call SAME selector.
pub(super) fn selected_recipe(cfg:&crate::serve::config::Gemma4Config,routing:bool)->acquisition::ConfigRecipe<'static> {
    let mut r=recipe(cfg);
    if routing {r.route="gemma-ngram-injected-native-routing1511-v1";}r
}
fn reviewed_workload(p:&Packet)->Result<Workload> {
    let raw=acquisition::checked_file(&p.greedy_workload,1024*1024)?;
    let work:Workload=serde_json::from_slice(&raw)?;work.validate().map_err(anyhow::Error::msg)?;
    let digest=acquisition::tokens_digest(&work)?;
    ensure!(work.family==Family::Gemma4 && work.vocab_size==262144 && digest==p.identity.token_vectors_sha256 &&
        digest!="922bec263c3c0159228b05a854b30f09d97403f1be442188d911388e1f868e3b","NEW independently frozen greedy144, not forced922");
    ensure!(p.independent_dataset_review.sha256=="9b09720893cf704c8cb9afa793a398da884f3c984d69892d194b5e501ae2f93c",
        "exact independently accepted NEW dataset receipt");
    let review:Value=serde_json::from_slice(&acquisition::checked_file(&p.independent_dataset_review,1024*1024)?)?;
    ensure!(review["schema"]=="hf2q.q1.gemma-final-matrix-dataset-review.v1" &&
        review["verdict"]=="AcceptedOneModelAssistedGreedy144Dataset" && review["reviewer"]=="q5-review" &&
        review["workload"]["path"].as_str()==p.greedy_workload.path.to_str() && review["workload"]["sha256"]==p.greedy_workload.sha256 &&
        review["full144_token_vectors_sha256"]==digest && review["checked_rows"]==68 &&
        review["payload_bytes_audited"]==71303168u64 && review["all_payloads_pages_finite_checked"]==true &&
        review["actual_gpu_choice_chain_independently_verified"]==true && review["first16_unique_nonEOS_each_lane"]==true &&
        review["carry_consumed_as145th_input"]==false && review["original_preparation_flags_modified"]==false &&
        review["preparation_provenance"]=="WorkloadPreparationNotComparable" &&
        review["preparation_source"]["model_sha256"]==p.identity.model_sha256 &&
        review["preparation_source"]["tokenizer_sha256"]==p.identity.tokenizer_sha256 &&
        review["preparation_source"]["template_sha256"]==p.identity.template_sha256,
        "independent actual68-row dataset review, not helper eligibility/selfapproval");
    let root=p.greedy_workload.path.parent().context("dataset output root")?;
    let preparation_binding:FileBinding=serde_json::from_value(json!({"path":root.join("preparation.json"),"sha256":review["preparation_sha256"]}))?;
    let trace_binding:FileBinding=serde_json::from_value(json!({"path":root.join("trace.json"),"sha256":review["trace_sha256"]}))?;
    let preparation:Value=serde_json::from_slice(&acquisition::checked_file(&preparation_binding,1024*1024)?)?;
    let trace:Value=serde_json::from_slice(&acquisition::checked_file(&trace_binding,128*1024*1024)?)?;
    ensure!(preparation["schema"]=="hf2q.q1.gemma-model-assisted-greedy144.v1" &&
        preparation["eligible_pending_independent_review"]==true && preparation["proposed_workload_json_sha256"]==p.greedy_workload.sha256 &&
        preparation["proposed_workload_token_vectors_sha256"]==digest && preparation["actual_checked_forward_count"]==68 &&
        preparation["decisions"].as_array().is_some_and(|v|v.len()==68) &&
        trace["coverage"]["captured"].as_array().is_some_and(|v|v.len()==68) &&
        trace["acquisition"]["provenance"]=="WorkloadPreparationNotComparable","review joins actual preparation, originals unchanged");
    Ok(work)
}
#[test]
#[ignore="root only selected Gemma model: new reviewed greedy dataset, genuine ngram transaction"]
fn q1_gemma_actual_ngram_transaction_trace(){run().unwrap();}
fn run()->Result<()> {
    use crate::serve::api::engine::{LoadOptions,LoadedModel};
    ensure!(cfg!(target_endian="little"),"LE artifact encoding");
    let packet=std::env::var_os("HF2Q_Q1_GEMMA_NGRAM_PACKET").context("explicit Gemma ngram packet")?;
    let mut raw=Vec::new();File::open(packet)?.take(1024*1024+1).read_to_end(&mut raw)?;ensure!(raw.len()<=1024*1024,"packet bound");
    let p:Packet=serde_json::from_slice(&raw)?;preflight(&p)?;let workload=reviewed_workload(&p)?;
    let acquisition=acquisition::Scope::begin(&p.acquisition,&p.identity,&workload,&p.model,&p.output,&raw)?;
    let header=acquisition::open_model(&p.model)?;ensure!(header.metadata_string("general.architecture")==Some("gemma4"),"explicit Gemma");
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&header)?;cfg.max_position_embeddings=262144;
    ensure!(cfg.num_hidden_layers==30 && cfg.hidden_size==2816 && cfg.num_experts==128 && cfg.top_k_experts==8 &&
        cfg.sliding_window>=148 && cfg.vocab_size==262144 && cfg.num_attention_heads==16 &&
        cfg.num_key_value_heads==8 && cfg.num_global_key_value_heads==2 && cfg.head_dim==256 && cfg.global_head_dim==512 &&
        (0..30).all(|l|cfg.is_full_attention(l)==((l+1)%6==0)),"selected MoE geometry before model");
    acquisition.check_config(&p.identity,&selected_recipe(&cfg,p.routing_witness.is_some()))?;drop(header);
    // Source maximum:16 snapshots (4 epochs×4 phases), each <=148×(3+8)×56320 plus norms,
    // plus <=84 full logits rows. This upper is conservative, not a physical-memory cap.
    let planned=planned_payload_upper();
    ensure!(planned<PAYLOAD && 16*1050+84<20000,"finite complete role/payload bounds before load");
    if p.routing_witness.is_some(){ensure!(planned+crate::adr063_invariance::gemma_routing_witness::PAYLOAD_BYTES+
        crate::adr063_invariance::gemma_routing_witness::METADATA_BYTES+crate::adr063_invariance::gemma_routing_witness::CPU_LEDGER_BYTES<=p.payload_limit,
        "routing+metadata+CPU ledger must join existing pre-load case admission");}
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let opts=LoadOptions {model_path:p.model.clone(),tokenizer_path:None,config_path:None,dwq_overlay_path:None,
        kv_persist_dir:None,kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None};
    let LoadedModel::Gemma(mut m)=LoadedModel::load_with_context(&opts,Some(262144))? else {anyhow::bail!("selected Gemma");};
    acquisition.check_config(&p.identity,&selected_recipe(&m.config,p.routing_witness.is_some()))?;
    ensure!(m.eos_token_ids==[1,106],"frozen actual EOS");
    let observed=acquisition.observed(m.ctx.executor.device(),&m as *const _ as usize as u64,3)?;
    let scope=Scope::install(&p.output,json!({"schema":"hf2q.q1.gemma-ngram-transaction.v1","identity":p.identity,
        "acquisition":observed,"packet_sha256":crate::adr063_invariance::store::hash(&raw),"recipe":selected_recipe(&cfg,p.routing_witness.is_some()),
        "greedy_workload":p.greedy_workload.sha256,"dataset_review":p.independent_dataset_review.sha256,
        "configured_context":262144,"configured_output_allowance":8192,"serial_owner_epochs":4,
        "not_mounted_slots":true,"physical_batch4_claim":false,"planned_payload_upper":planned}))?;
    let _routing=crate::adr063_invariance::gemma_routing_witness::install(m.ctx.executor.device(),&p.output,&p.identity,&observed,
        crate::adr063_invariance::gemma_routing_witness::Mode::Transaction,p.routing_witness.as_ref())?;

    let result=(||->Result<()> {
        for lane in 0..4 {
            // Checked prior owner already returned. Clear global active banks, no physical-refund claim.
            m.weights.hybrid_kv=None;m.weights.dense_kvs=None;m.weights.leg_hb_encoded=None;
            m.weights.hybrid_kv_snapshot_for_lcp=None;m.weights.dense_kvs_snapshot_for_lcp=None;
            for c in &mut m.weights.kv_caches {c.write_pos=0;c.seq_len=0;}
            m.weights.decode_step=0;scope.owner(lane,&workload.lanes[lane])?;
            let mut output=crate::inference::spec_decode::ngram_orchestrator::dispatch_ngram_generate(
                &mut m.weights,&workload.lanes[lane][..128],8192,3,1,3,&m.eos_token_ids,&mut m.ctx)?;
            with(|a|{transition(a.owner.as_ref().context("owner")?.phase,Phase::Committed)})?.context("one actual transaction absent")?;
            let mut carry=0;
            while output.len()<=144 {
                let pos=output.len()-1;ensure!((129..144).contains(&pos),"actual posttransaction cursor/input");
                let input=*output.last().context("unprocessed actual token")?;ensure!(!m.eos_token_ids.contains(&input),"actual EOS: no padding or forced continuation");
                crate::adr063_invariance::gemma_routing_witness::serial(
                    crate::adr063_invariance::gemma_routing_witness::Namespace::T5Continuation,lane,pos,1)?;
                let next=m.weights.forward_decode(input,pos,&mut m.ctx,&mut None)?;
                crate::adr063_invariance::gemma_routing_witness::end_checked()?;
                with(|a|a.journal.tensor(json!({"owner_epoch":lane,"phase":"ordinary_postrollback_logits","position":pos,
                    "actual_input_id":input,"actual_gpu_next_id":next}),&m.weights.activations.logits))?;
                if pos==143 {carry=next;break;}output.push(next);
            }
            scope.observed_end(&m.weights,&output,carry)?;
        }Ok(())
    })();
    acquisition.check_unchanged()?;
    with(|a|a.journal.event("execution_result",json!({"failure":result.as_ref().err().map(ToString::to_string),
        "zero_acceptance_is_valid_evidence":true,"ordinary_proposer_quality_qualified":false,"whole_q1_qualified":false})))?;
    scope.finish(if result.is_ok(){"CompleteDiagnosticPendingIndependentReview"}else{"FailedOrIncomplete"})?;result?;
    crate::adr063_invariance::gemma_routing_witness::finish()?;
    println!("Q1_GEMMA_ACTUAL_NGRAM_TRANSACTIONS_COMPLETE owners=4 real_transactions=4 injected_proposals=true t5_qualified=false");Ok(())
}
