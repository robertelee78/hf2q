//! Full original four-lane coding W. Neither extents nor turn coverage come
//! from candidate K4 output. Raw request bytes (including IDs/tool paths) stay bound.
use super::*;
use serde_json::Value;
use super::frozen_params::{FrozenParams, TokenTable};
use std::sync::Arc;
pub(crate) use super::original_witness::OriginalWitness;

pub(crate) const VARIANT_SHA: &str = "b6a671e3c52adc8bb0bbdd62584973f3720fd97c7a92b2c2f4eaadbd20849991";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SourceTurn { pub(crate) turn:usize, pub(crate) request:acquisition::FileBinding,
    pub(crate) receipt:acquisition::FileBinding, pub(crate) response:acquisition::FileBinding,
    pub(crate) original_witness:Option<acquisition::FileBinding> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SourceLane { pub(crate) lane:usize, pub(crate) summary:acquisition::FileBinding,
    pub(crate) turns:Vec<SourceTurn> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Corpus { pub(crate) family:Family, pub(crate) case:acquisition::FileBinding,
    pub(crate) variant_fixture:acquisition::FileBinding, pub(crate) lanes:Vec<SourceLane> }
#[derive(Default)]
pub(crate) struct OriginalCorpus {
    pub(crate) requests:Vec<Vec<Vec<u8>>>,
    pub(crate) witnesses:Vec<Vec<Arc<OriginalWitness>>>,
}
#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ForwardBoundary {
    pub(crate) start:usize,pub(crate) input:Vec<u32>,pub(crate) logit_rows:usize,
    pub(crate) hidden_rows:usize,pub(crate) route:String,
}
#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct Prepared {
    pub(crate) request_sha256:String, pub(crate) prompt:Vec<u32>, pub(crate) params:FrozenParams,
    pub(crate) registration_architecture:String,
    pub(crate) cached_tokens:Option<usize>,pub(crate) transactions:Vec<ForwardBoundary>,
}
#[derive(Deserialize, Serialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Terminal {
    pub(crate) request_sha256:String, pub(crate) slot:u32,
    pub(crate) decisions:Vec<u32>, pub(crate) generated:Vec<u32>, pub(crate) finish_reason:String,
    pub(crate) native_step:usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReferenceTurn { pub(crate) turn:usize, pub(crate) prepared:acquisition::FileBinding,
    pub(crate) terminal:acquisition::FileBinding, pub(crate) response:acquisition::FileBinding }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReferenceLane { pub(crate) lane:usize, pub(crate) turns:Vec<ReferenceTurn> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Script {
    pub(crate) schema:String, pub(crate) corpus:Corpus,
    pub(crate) reference_identity:Identity, pub(crate) reference_route:String,
    pub(crate) token_table:Option<acquisition::FileBinding>, pub(crate) lanes:Vec<ReferenceLane>,
    pub(crate) prefill_chunk:usize,
    pub(crate) anchor_bytes_per_slot_max:u64,
}
pub(crate) struct Turn { pub(crate) prepared:Prepared,
    pub(crate) terminal:Terminal }
pub(crate) struct Validated {
    pub(crate) script:Script, pub(crate) turns:Vec<Vec<Turn>>,
    pub(crate) originals:OriginalCorpus,
}
fn json(binding:&acquisition::FileBinding)->Result<Value> {
    Ok(serde_json::from_slice(&acquisition::checked_file(binding,64*1024*1024)?)?)
}
impl Corpus {
    pub(crate) fn validate(&self, family:Family)->Result<OriginalCorpus> {
        let requests = self.validate_archived_requests(family)?;
        let mut witnesses = Vec::with_capacity(4);
        for (lane, source) in self.lanes.iter().enumerate() {
            let mut turns = Vec::with_capacity(source.turns.len());
            for (turn, item) in source.turns.iter().enumerate() {
                let binding = item.original_witness.as_ref().with_context(||
                    format!("AcquisitionBlocked: original native witness missing for lane{lane}/turn{turn}"))?;
                let request: Value = serde_json::from_slice(&requests[lane][turn])?;
                turns.push(Arc::new(OriginalWitness::load(binding, &item.request.sha256,
                    &item.response.sha256, family, request["stream"] == true, &json(&item.receipt)?,
                    &acquisition::checked_file(&item.response, 64*1024*1024)?)?));
            }
            witnesses.push(turns);
        }
        Ok(OriginalCorpus { requests, witnesses })
    }
    /// Archived HTTP inputs/origin only. This never attests native observations.
    pub(crate) fn validate_archived_requests(&self, family:Family)->Result<Vec<Vec<Vec<u8>>>> {
        ensure!(family == self.family,"original artifact family differs");
        archived_requests(family,&self.case,Some(&self.variant_fixture),&self.lanes)
    }
}
/// Shared HTTP origin validation; `None` means the separately declared append
/// corpus, not a relaxation of historical divergent native-witness authority.
pub(crate) fn archived_requests(family:Family,case_binding:&acquisition::FileBinding,
    fixture:Option<&acquisition::FileBinding>,lanes:&[SourceLane])->Result<Vec<Vec<Vec<u8>>>> {
        ensure!(matches!(family,Family::Qwen38|Family::Qwen36),"wrong original artifact family");
        let case=json(case_binding)?; let workload=&case["workload"];
        ensure!(case["family"]==match family {Family::Qwen38=>"qwen38",Family::Qwen36=>"qwen36",_=>unreachable!()},
            "case/artifact family differs");
        ensure!(case["schema"]=="hf2q.q5.calibration-case.v1" && workload["slots"]==4
            && workload["context_per_slot"]==262144 && workload["anchors_per_slot"]==4,
            "full original262144/four-slot/K4 case required, not matrix capacity");
        ensure!(workload["client"].as_str().is_some_and(|s|s.contains("qwen_n4_live_smoke.py")
            &&s.contains("divergent-edit-v1")==fixture.is_some()),"declared append/divergent coding input differs");
        let marker=if let Some(fixture)=fixture {
            ensure!(fixture.sha256==VARIANT_SHA,"unpinned edit fixture");
            Some(String::from_utf8(acquisition::checked_file(fixture,4096)?)?)
        }else{None};
        ensure!(lanes.len()==4,"all four original lanes required");
        let mut requests=Vec::with_capacity(4); let mut cohort_run=None;
        for (lane,source) in lanes.iter().enumerate() {
            ensure!(source.lane==lane,"four original lanes must be exactly0..3");
            let summary=json(&source.summary)?;
            ensure!(summary["schema"]=="hf2q.qwen-n4-agent-result.v1" && summary["profile"]=="adr063-qwen-n4-live-v1"
                && summary["agent_id"]==format!("agent-{lane}") && summary["diagnostic_slots"]==4,
                "not the original four-lane receipt");
            let run=summary["run_id"].as_str().context("original runID missing")?.to_owned();
            if let Some(expected)=&cohort_run { ensure!(expected==&run,"mixed original cohorts"); } else {cohort_run=Some(run);}
            ensure!(summary["status"]=="ProtocolCompleteAwaitingIndependentPatchReview","original source workload incomplete");
            if fixture.is_some() {
                ensure!(summary["workload_variant"]["id"]=="divergent-edit-v1"
                    &&summary["workload_variant"]["fixture_sha256"]==VARIANT_SHA
                    &&summary["workload_variant"]["edit_turns"]==serde_json::json!([3,5])
                    &&summary["workload_variant"]["keep_lines"]==16,"actual divergent turns3/5 required");
            }else{
                ensure!(summary["workload_variant"].is_null(),"append origin contains undeclared variant");
            }
            let receipts=summary["turns"].as_array().context("full original turns missing")?;
            let tools=summary["tools"].as_array().context("original tools missing")?;
            let repair=summary["repair_turns"].as_u64().unwrap_or(0);
            ensure!(repair==u64::from(family==Family::Qwen36),"original Qwen36 repair1/Qwen38 repair0 contract changed");
            ensure!(receipts.len()==tools.len()+1 && (tools.len()==6 || repair==1&&tools.len()==8)
                && source.turns.len()==receipts.len(),"truncated original7-or9-request transcript");
            if fixture.is_some() {
                let edits=summary["workload_variant_edits"].as_array().context("original edits missing")?;
                ensure!(edits.len()==2&&edits[0]["turn"]==3&&edits[1]["turn"]==5,"actual divergent edit audit missing");
            }else{
                ensure!(summary["workload_variant_edits"].is_null()||summary["workload_variant_edits"]==serde_json::json!([]),
                    "append origin contains divergent edits");
            }
            let mut lane_requests=Vec::with_capacity(receipts.len()); let mut previous:Option<Value>=None;
            for (turn,source_turn) in source.turns.iter().enumerate() {
                ensure!(source_turn.turn==turn && receipts[turn]["turn"]==turn,"original turn IDs must be complete zero-based order");
                let raw=acquisition::checked_file(&source_turn.request,32*1024*1024)?;
                let req:Value=serde_json::from_slice(&raw)?; let receipt=json(&source_turn.receipt)?;
                ensure!(receipt==receipts[turn] && receipt["http_status"]==200
                    && receipt["request_sha256"]==hash(&raw) && receipt["request_bytes"]==raw.len(),"source raw-request/receipt binding differs");
                let response=acquisition::checked_file(&source_turn.response,64*1024*1024)?;
                ensure!(receipt["response_sha256"]==hash(&response) && receipt["response_bytes"]==response.len(),"source response binding differs");
                ensure!(req["max_completion_tokens"]==8192 && req["tool_choice"]=="auto"
                    && req["hf2q_overflow_policy"]=="reject" && req["stream"]==(turn%2==1),"original request controls changed");
                let messages=req["messages"].as_array().context("full source messages missing")?;
                if let Some(marker)=marker.as_deref().filter(|_|turn==3||turn==5) {
                    let marker=marker.trim();
                    let previous=previous.as_ref().context("divergent source predecessor missing")?;
                    let mut prefix=previous["messages"].as_array().context("previous messages missing")?.clone();
                    let target=turn-2;
                    let index=prefix.iter().enumerate().filter(|(_,m)|m["role"]=="tool").nth(target)
                        .map(|(i,_)|i).context("original divergent tool result missing")?;
                    let old=prefix[index]["content"].as_str().context("non-text divergent source result")?;
                    let edited=old.lines().take(16).chain(std::iter::once(marker)).collect::<Vec<_>>().join("\n");
                    ensure!(edited!=old,"vacuous divergent edit");
                    prefix[index]["content"]=Value::String(edited);
                    ensure!(messages.len()>prefix.len() && messages[..prefix.len()]==prefix,"not exact original16-line divergence transform");
                }
                previous=Some(req); lane_requests.push(raw);
            }
            requests.push(lane_requests);
        }
        Ok(requests)
    }
impl Script {
    pub(crate) fn validate_bytes(raw:&[u8],family:Family)->Result<Validated> {
        let script:Self=serde_json::from_slice(raw)?;
        ensure!(script.schema=="hf2q.c2.full-original-frozen-script.v1","not a full-W script");
        let originals=script.corpus.validate(family)?;
        let requests=&originals.requests;
        ensure!(script.lanes.len()==4 && script.prefill_chunk>0 && script.prefill_chunk<=8192,
            "complete four-lane/bounded actual prefill schedule required");
        ensure!(script.anchor_bytes_per_slot_max>0 && script.anchor_bytes_per_slot_max<=16*1024*1024*1024,
            "source-bound production capture/cache budget missing");
        let case=json(&script.corpus.case)?;
        ensure!(case["workload"]["anchor_bytes_per_slot_max"]==script.anchor_bytes_per_slot_max,
            "capture quota differs from source-bound case");
        ensure!(script.reference_identity.family==family
            && script.reference_identity.model_sha256==match family {Family::Qwen38=>Q38,Family::Qwen36=>Q36,_=>unreachable!()}
            && script.reference_route=="actual-production-HTTP-OFF-canonical",
            "canonical decisions must be independently captured real HTTP target output");
        let token_table=script.token_table.as_ref().map(TokenTable::load).transpose()?;
        let mut turns=Vec::with_capacity(4);
        for (lane,reference) in script.lanes.iter().enumerate() {
            ensure!(reference.lane==lane && reference.turns.len()==requests[lane].len(),"equal-truncation/omitted-lane reference");
            let mut lane_turns=Vec::with_capacity(reference.turns.len());
            for (turn,item) in reference.turns.iter().enumerate() {
                ensure!(item.turn==turn,"missing original reference turn");
                let prepared:Prepared=serde_json::from_value(json(&item.prepared)?)?;
                let terminal:Terminal=serde_json::from_value(json(&item.terminal)?)?;
                let response=acquisition::checked_file(&item.response,64*1024*1024)?;
                ensure!(!response.is_empty(),"real canonical HTTP response missing");
                let request=&requests[lane][turn]; let sha=hash(request);
                ensure!(prepared.request_sha256==sha && terminal.request_sha256==sha,"new/normalized request bytes substituted");
                ensure!(!prepared.prompt.is_empty() && prepared.prompt.len()+8192<=262144
                    && prepared.params.max_tokens==8192 && !terminal.generated.is_empty()
                    && terminal.generated.len()<=8192 && terminal.decisions.starts_with(&terminal.generated)
                    && terminal.decisions.len()>=terminal.generated.len() && terminal.decisions.len()<=terminal.generated.len()+1
                    && matches!(terminal.finish_reason.as_str(),"stop"|"length"),"incomplete actual canonical admission/terminal tape");
                ensure!(prepared.params.stable_prompt_prefix_tokens.is_some_and(|n|n>0&&n<prepared.prompt.len()),
                    "actual original generation-cue checkpoint boundary missing");
                let req:Value=serde_json::from_slice(request)?;
                let original=&originals.witnesses[lane][turn];
                ensure!(original.producer_identity.tokenizer_sha256==script.reference_identity.tokenizer_sha256
                    &&original.producer_identity.template_sha256==script.reference_identity.template_sha256,
                    "AcquisitionBlocked: original witness tokenizer/template identity differs");
                original.require_response(&response,req["stream"]==true,&prepared,&terminal)?;
                ensure!(prepared.cached_tokens.is_some_and(|n|n<=prepared.prompt.len())
                    &&!prepared.transactions.is_empty(),"actual admitted cache/checked transaction boundaries missing");
                ensure!(prepared.transactions.iter().all(|call|!call.input.is_empty()
                    &&call.input.len()<=8192&&call.start.checked_add(call.input.len()).is_some_and(|end|end<=262144)
                    &&call.logit_rows>0&&call.logit_rows<=call.input.len()
                    &&call.hidden_rows<=call.input.len()&&!call.route.is_empty()),
                    "invalid actual forward transaction boundary");
                for index in 0..terminal.decisions.len() {
                    let position=prepared.prompt.len()-1+index;
                    let covered=prepared.transactions.iter().any(|call| {
                        if call.logit_rows==0||call.logit_rows>call.input.len(){return false}
                        let row_start=call.start+call.input.len()-call.logit_rows;
                        if position<row_start||position>=call.start+call.input.len(){return false}
                        call.input[..=position-call.start].iter().enumerate().all(|(offset,&token)| {
                            let p=call.start+offset;
                            let expected=if p<prepared.prompt.len(){prepared.prompt.get(p)}
                                else{terminal.decisions.get(p-prepared.prompt.len())};
                            expected.is_some_and(|&expected|expected==token)
                        })
                    });
                    ensure!(covered,"independent canonical decision{index} lacks actual committed forward transaction");
                }
                let tools:Vec<crate::serve::api::schema::Tool>=serde_json::from_value(req["tools"].clone())?;
                prepared.params.restore(token_table.as_ref(),&tools)?;
                lane_turns.push(Turn {prepared,terminal});
            }
            turns.push(lane_turns);
        }
        Ok(Validated {script,turns,originals})
    }
}

