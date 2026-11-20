//! Prospective fixed-input numerical authority. Historical HTTP bodies are inputs,
//! never a claim that their embedded tool results were produced by this execution.
use super::*;
use super::frozen_script::{SourceLane, SourceTurn, Prepared, Terminal};
use super::frozen_params::TokenTable;
use serde_json::Value;
use std::collections::BTreeSet;

pub(crate) const INPUT_SCHEMA: &str = "hf2q.c2.fresh-fixed-inputs.v1";
pub(crate) const CAPTURE_SCHEMA: &str = "hf2q.c2.fresh-fixed-input-capture.v1";
pub(crate) const REGISTRATION_SCHEMA: &str = "hf2q.c2.fresh-fixed-input-registration.v1";
const ORIGINAL_INPUT_INVENTORY_SHA256: &str =
    "7cdeda4ff353a52631c49ee2b2556d27dcd28a49b6df694a7074dd9e271f8745";

fn original_input_inventory(binding: &acquisition::FileBinding) -> Result<Value> {
    ensure!(binding.sha256 == ORIGINAL_INPUT_INVENTORY_SHA256,
        "fixed-input inventory is not the independently approved original116");
    let inventory: Value = serde_json::from_slice(&acquisition::checked_file(binding, 4*1024*1024)?)?;
    ensure!(inventory["schema"] == "hf2q.private.c2-original-http-input-inventory.v1"
        && inventory["purpose"] == "historical fixed-input origin only",
        "independent original input inventory required");
    Ok(inventory)
}
/// Fresh runtime identities bind the actual published dependency compiled into
/// this executable. Historical C2 identities use their separate immutable role.
fn compiled_native_checksum()->Result<&'static str> {
    static CHECKSUM:std::sync::LazyLock<Option<&'static str>>=std::sync::LazyLock::new(|| {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"),"/Cargo.lock")).split("[[package]]").find_map(|package| {
            let field=|name:&str|package.lines().find_map(|line|line.strip_prefix(name).and_then(|s|s.strip_suffix('"')));
            (field("name = \"")==Some("mlx-native")&&field("version = \"")==Some(env!("HF2Q_MLX_NATIVE_VERSION")))
                .then(||field("checksum = \"")).flatten()
        })
    });
    (*CHECKSUM).context("compiled published native checksum missing")
}
pub(crate) fn validate_current_identity(identity:&Identity)->Result<()> {
    identity.validate().map_err(anyhow::Error::msg)?;
    ensure!(identity.native_checksum==compiled_native_checksum()?,"fresh native checksum differs from compiled published native pin");
    Ok(())
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputTurn {
    pub turn: usize, pub request: acquisition::FileBinding,
    pub receipt: acquisition::FileBinding, pub response: acquisition::FileBinding,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputLane {
    pub lane: usize, pub summary: acquisition::FileBinding, pub turns: Vec<InputTurn>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Inputs {
    pub schema: String, pub family: Family, pub case: acquisition::FileBinding,
    pub workload_kind:String, pub variant_fixture: Option<acquisition::FileBinding>, pub lanes: Vec<InputLane>,
    pub origin_inventory:acquisition::FileBinding, pub origin:String,
}
impl Inputs {
    pub(crate) fn validate(&self, family: Family) -> Result<Vec<Vec<Vec<u8>>>> {
        ensure!(self.schema == INPUT_SCHEMA && self.family == family, "fresh fixed-input schema/family differs");
        ensure!(matches!((self.workload_kind.as_str(),self.variant_fixture.is_some()),("append",false)|("divergent",true)),
            "explicit source-bound append/divergent input kind required");
        let inventory = original_input_inventory(&self.origin_inventory)?;
        let family_name=match family {Family::Qwen38=>"Qwen38",Family::Qwen36=>"Qwen36",_=>anyhow::bail!("unsupported fresh family")};
        let variant_name=match (family,self.workload_kind.as_str()) {
            (Family::Qwen38,"append")=>"append",(Family::Qwen38,"divergent")=>"divergent",
            (Family::Qwen36,"append")=>"append-repair1",(Family::Qwen36,"divergent")=>"divergent-repair1",
            _=>anyhow::bail!("unsupported exact original inventory variant"),
        };
        let mut origins=inventory["origins"].as_array().context("inventory origins missing")?.iter()
            .filter(|origin|origin["family"]==family_name&&origin["variant"]==variant_name&&origin["origin"]==self.origin);
        let origin=origins.next().context("selected fixed-input origin absent from independent inventory")?;
        ensure!(origins.next().is_none()&&origin["lanes"].as_array().is_some_and(|lanes|lanes.len()==4),
            "ambiguous/incomplete fixed-input inventory origin");
        for (lane,input) in self.lanes.iter().enumerate() {
            ensure!(origin["turn_counts"][lane].as_u64()==Some(input.turns.len() as u64)
                &&origin["lanes"][lane]["lane"].as_u64()==Some(input.lane as u64)
                &&origin["lanes"][lane]["summary"]==serde_json::to_value(&input.summary)?,
                "fixed-input manifest changed independent per-lane extent/origin");
            for (turn,item) in input.turns.iter().enumerate() {
                let original=&origin["lanes"][lane]["turns"][turn];
                ensure!(original["turn"].as_u64()==Some(item.turn as u64)
                    &&original["request"]==serde_json::to_value(&item.request)?
                    &&original["response"]==serde_json::to_value(&item.response)?
                    &&original["receipt"]==serde_json::to_value(&item.receipt)?,
                    "fixed-input manifest substituted an original request/response/receipt binding");
            }
        }
        ensure!(self.lanes.len() == 4 && self.lanes.iter().enumerate().all(|(lane, input)|
            input.lane == lane && !input.turns.is_empty() && input.turns.iter().enumerate().all(|(turn, item)| item.turn == turn)),
            "complete ordered per-lane archived fixed-input manifest required");
        // Reuse only the archived HTTP origin checks. This function does NOT load
        // OriginalWitness or attest any historical native token identity.
        let lanes=self.lanes.iter().map(|lane| SourceLane {lane:lane.lane,summary:lane.summary.clone(),
            turns:lane.turns.iter().map(|turn|SourceTurn {turn:turn.turn,request:turn.request.clone(),
                receipt:turn.receipt.clone(),response:turn.response.clone(),original_witness:None}).collect()}).collect::<Vec<_>>();
        super::frozen_script::archived_requests(family,&self.case,self.variant_fixture.as_ref(),&lanes)
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StopSelection { pub token: u32, pub cause: String }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Observation {
    pub ticket: String, pub generation: u64,
    pub prepared: Prepared, pub terminal: Terminal,
    pub stop_selection: Option<StopSelection>,
    pub request: acquisition::FileBinding, pub response: acquisition::FileBinding,
    pub response_headers: acquisition::FileBinding,
    pub server_request:acquisition::FileBinding, pub request_headers:acquisition::FileBinding,
    pub server_response:acquisition::FileBinding,
    pub server_response_sha256: String, pub server_response_bytes: usize,
    pub http_finish_reason: String, pub completion_counter: String,
    pub published_completion_tokens: usize,
    pub acquisition: acquisition::Acquisition,
    pub actual_routes: Vec<String>, pub raw_decisions: Vec<u32>,
    pub gpu_frozen_selector_used: bool,
    pub actual_gpu_argmax:Vec<(usize,u32)>,
}
fn require_request_ticket(headers:&Value,ticket:&str)->Result<()> {
    ensure!(headers["ticket"]==ticket,"redundant ingress ticket differs");
    let mut actual=None;
    for header in headers["headers"].as_array().context("captured request headers missing")? {
        let row=header.as_array().context("captured header is not a pair")?;
        ensure!(row.len()==2,"captured header is not a pair");
        let name=row[0].as_str().context("captured header name missing")?;
        if name.eq_ignore_ascii_case("x-hf2q-c2-ticket") {
            ensure!(actual.is_none(),"duplicate captured request ticket header");
            actual=Some(row[1].as_array().context("captured ticket bytes missing")?);
        }
    }
    let actual=actual.context("actual captured request ticket header missing")?;
    ensure!(actual.len()==ticket.len()&&actual.iter().zip(ticket.bytes()).all(|(value,byte)|
        (byte==b'\t'||(32..=126).contains(&byte))&&value.as_u64()==Some(u64::from(byte))),
        "actual captured request ticket bytes differ");
    Ok(())
}
impl Observation {
    pub(crate) fn validate(&self, raw: &[u8], stream: bool, identity: &Identity, unforced: bool) -> Result<()> {
        let prepared = &self.prepared; let terminal = &self.terminal;
        ensure!(self.generation > 0 && !self.ticket.is_empty() && self.ticket.len() <= 192,
            "missing request ticket/slot generation");
        ensure!(self.request.sha256 == hash(raw) && acquisition::checked_file(&self.request, 256*1024)? == raw
            && self.server_request.sha256==self.request.sha256 && acquisition::checked_file(&self.server_request,256*1024)?==raw
            && prepared.request_sha256 == self.request.sha256 && terminal.request_sha256 == self.request.sha256,
            "native observation changed exact fixed request bytes");
        let request_headers:Value=serde_json::from_slice(&acquisition::checked_file(&self.request_headers,1024*1024)?)?;
        require_request_ticket(&request_headers,&self.ticket)?;
        ensure!(!prepared.prompt.is_empty() && prepared.prompt.len()+8192 <= 262144 && prepared.params.max_tokens == 8192
            && terminal.slot < 4 && terminal.generated.len() <= 8192 && !terminal.decisions.is_empty()
            && terminal.decisions.starts_with(&terminal.generated)
            && terminal.decisions.len() <= terminal.generated.len()+1
            && matches!(terminal.finish_reason.as_str(), "stop"|"length") && terminal.native_step > 0,
            "incomplete prepared/generated/decision/native-terminal tape");
        let response = acquisition::checked_file(&self.response, 64*1024*1024)?;
        ensure!(self.response.sha256 == self.server_response_sha256 && self.server_response.sha256==self.response.sha256
            &&acquisition::checked_file(&self.server_response,64*1024*1024)?==response&&response.len() == self.server_response_bytes,
            "client/server raw response differs");
        let wire = super::original_witness::wire_terminal(&response, stream)?;
        require_terminal(terminal,self.stop_selection.as_ref(),&self.completion_counter,self.published_completion_tokens,&wire.finish_reason)?;
        ensure!(wire.prompt_tokens == prepared.prompt.len() && wire.completion_tokens == self.published_completion_tokens
            && wire.finish_reason == self.http_finish_reason, "HTTP/native usage or terminal mismatch");
        let headers: Value = serde_json::from_slice(&acquisition::checked_file(&self.response_headers, 1024*1024)?)?;
        ensure!(headers["status"] == 200, "HTTP error is not a complete numerical observation");
        ensure!(prepared.cached_tokens.is_some_and(|n| n <= prepared.prompt.len()) && !prepared.transactions.is_empty()
            && prepared.params.stable_prompt_prefix_tokens.is_some_and(|n| n > 0 && n < prepared.prompt.len()),
            "actual admission/forward/cue boundaries missing");
        for index in 0..terminal.decisions.len() {
            let position = prepared.prompt.len()-1+index;
            ensure!(prepared.transactions.iter().any(|call| {
                let Some(end) = call.start.checked_add(call.input.len()) else { return false };
                if call.input.is_empty() || call.input.len()>8192 || end>262144 || call.logit_rows==0
                    || call.logit_rows>call.input.len() || call.hidden_rows>call.input.len() || call.route.is_empty()
                    || position < end-call.logit_rows || position >= end { return false }
                call.input.iter().take(position-call.start+1).enumerate().all(|(offset, token)| {
                    let at=call.start+offset;
                    prepared.prompt.get(at).or_else(|| at.checked_sub(prepared.prompt.len())
                        .and_then(|i| terminal.decisions.get(i))) == Some(token)
                })
            }), "canonical decision lacks full consumed-window provenance");
        }
        ensure!(self.acquisition.actual_binary_sha256 == identity.binary_sha256
            && self.acquisition.required_model_opens > 0
            && self.acquisition.observed_model_opens >= self.acquisition.required_model_opens
            && self.acquisition.opened_model.inode > 0 && self.acquisition.opened_model.bytes > 0
            && self.acquisition.device_registry_id != 0 && self.acquisition.queue_identity != 0
            && self.acquisition.model_instance != 0
            && self.acquisition.provenance == "IndependentRootReviewedComposition",
            "actual source/native/opened-FD/nonce provenance missing");
        if unforced {
            ensure!(!self.gpu_frozen_selector_used && self.raw_decisions == terminal.decisions,
                "reference acquisition used a frozen selector");
        }
        Ok(())
    }
}
fn require_terminal(terminal:&Terminal,stop:Option<&StopSelection>,counter:&str,published:usize,http_finish:&str)->Result<()> {
    ensure!(matches!((terminal.finish_reason.as_str(),http_finish),
        ("stop","stop"|"tool_calls")|("length","length")),"HTTP/native finish reasons contradict");
    ensure!(terminal.decisions.len()<=8192&&(terminal.finish_reason!="length"||terminal.generated.len()==8192),
        "incomplete native allowance terminal; fixed8192 budget cannot be silently shortened");
    match stop {
        Some(stop)=>ensure!(matches!(stop.cause.as_str(),"eos"|"grammar")&&terminal.finish_reason=="stop"
            &&terminal.decisions.len()==terminal.generated.len()+1&&terminal.decisions.last()==Some(&stop.token),
            "actual stopping selection disagrees with native terminal tape"),
        None=>ensure!(terminal.decisions==terminal.generated,"suppressed native terminal selection was not observed"),
    }
    ensure!(counter=="generated_tokens"&&published==terminal.generated.len(),
        "native publication counter mismatch; no trimming/padding");
    Ok(())
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Lane { pub lane: usize, pub observations: Vec<acquisition::FileBinding> }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Capture {
    pub schema: String, pub inputs: acquisition::FileBinding, pub owner_decision: acquisition::FileBinding,
    pub identity: Identity, pub packet: acquisition::FileBinding,
    pub token_table: Option<acquisition::FileBinding>, pub lanes: Vec<Lane>,
    pub reference_route: String, pub prefill_chunk: usize,
    pub acquisition_nonce: String, pub complete: bool, pub unforced: bool,
    pub historical_native_identity_recovered: bool, pub whole_c2_pass: bool,
}
pub(crate) struct Validated {
    pub capture: Capture, pub inputs: Inputs, pub requests: Vec<Vec<Vec<u8>>>,
    pub observations: Vec<Vec<Observation>>,
}
impl Capture {
    pub(crate) fn validate(self, family: Family) -> Result<Validated> {
        validate_current_identity(&self.identity)?;
        ensure!(self.schema == CAPTURE_SCHEMA && self.identity.family == family && self.complete && self.unforced
            && !self.historical_native_identity_recovered && !self.whole_c2_pass
            && self.reference_route == "fresh-fixed-input-HTTP-OFF-unforced-K4"
            && self.prefill_chunk > 0 && self.prefill_chunk <= 8192,
            "fresh independent complete unforced acquisition required");
        ensure!(self.identity.model_sha256 == if family==Family::Qwen38 {Q38} else {Q36}, "fresh artifact differs");
        let inputs: Inputs = serde_json::from_slice(&acquisition::checked_file(&self.inputs, 4*1024*1024)?)?;
        let requests = inputs.validate(family)?;
        ensure!(self.identity.token_vectors_sha256 == self.inputs.sha256, "capture identity must bind exact fixed-input manifest");
        let _: Value = serde_json::from_slice(&acquisition::checked_file(&self.owner_decision, 4*1024*1024)?)?;
        let packet: Value = serde_json::from_slice(&acquisition::checked_file(&self.packet, 4*1024*1024)?)?;
        ensure!(packet["schema"] == "hf2q.c2.fresh-fixed-input-http-packet.v1" && packet["mode"] == "capture"
            && packet["execution_approved"] == true && packet["family"] == serde_json::to_value(family)?
            && packet["identity"] == serde_json::to_value(&self.identity)? && packet["inputs"] == serde_json::to_value(&self.inputs)?
            && packet["owner_decision"] == serde_json::to_value(&self.owner_decision)?
            && packet["arm"] == "canonical" && packet["keep"] == 4 && packet["route"] == "Off"
            && packet["registration"].is_null() && packet["selected_cell"].is_null() && packet["expected"].is_null(),
            "capture packet is not its unforced producer binding");
        let acquisition_binding: acquisition::Binding = serde_json::from_value(packet["acquisition"].clone())?;
        let reviewed_model = acquisition::reviewed_model_identity(&acquisition_binding, &self.identity)?;
        let output: PathBuf = serde_json::from_value(packet["output"].clone())?;
        ensure!(output.is_absolute() && output.canonicalize()? == output,
            "capture output is not its canonical acquisition root");
        ensure!(self.packet.path == output.join("packet.json"),
            "capture packet copy is not in its owned output");
        ensure!(self.acquisition_nonce.len() == 64 && self.acquisition_nonce.bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "capture acquisition nonce is not canonical");
        let table = self.token_table.as_ref().map(TokenTable::load).transpose()?;
        ensure!(self.lanes.len() == 4, "fresh capture omitted a lane");
        let mut observations = Vec::with_capacity(4); let mut tickets = BTreeSet::new();
        let mut generations = BTreeSet::new();
        for (lane, source) in self.lanes.iter().enumerate() {
            ensure!(source.lane == lane && source.observations.len() == requests[lane].len(), "fresh capture truncated/reordered an independently bound lane");
            let mut turns = Vec::new();
            for (turn, binding) in source.observations.iter().enumerate() {
                let observed: Observation = serde_json::from_slice(&acquisition::checked_file(binding, 64*1024*1024)?)?;
                let raw = &requests[lane][turn]; let req: Value = serde_json::from_slice(raw)?;
                observed.validate(raw, req["stream"] == true, &self.identity, true)?;
                let acquired = &observed.acquisition;
                ensure!(acquired.packet_sha256 == self.packet.sha256 && acquired.output_root == output
                    && acquired.composition_sha256 == acquisition_binding.composition.sha256
                    && acquired.independent_review_sha256 == acquisition_binding.independent_review.sha256
                    && acquired.opened_model == reviewed_model,
                    "observation does not belong to its reviewed packet/composition/opened model");
                ensure!(tickets.insert(observed.ticket.clone()) && generations.insert(observed.generation)
                    && observed.acquisition.nonce == self.acquisition_nonce, "mixed/reused request/native generation or acquisition cohort");
                let tools:Vec<crate::serve::api::schema::Tool> = serde_json::from_value(req["tools"].clone())?;
                observed.prepared.params.restore(table.as_ref(), &tools)?;
                turns.push(observed);
            }
            observations.push(turns);
        }
        Ok(Validated { capture: self, inputs, requests, observations })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisterPacket {
    schema: String, capture: acquisition::FileBinding, independent_audit: acquisition::FileBinding,
    run_receipt: acquisition::FileBinding, output: PathBuf,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Registration {
    schema: String, pub capture: acquisition::FileBinding, independent_audit: acquisition::FileBinding,
    run_receipt: acquisition::FileBinding, pub reference_id: String,
}
fn registration_inputs(reg: &Registration) -> Result<Validated> {
    ensure!(reg.schema == REGISTRATION_SCHEMA && reg.reference_id == reg.capture.sha256, "fresh registration schema/content ID differs");
    let captured: Capture = serde_json::from_slice(&acquisition::checked_file(&reg.capture, 32*1024*1024)?)?;
    let family=captured.identity.family;
    let validated=captured.validate(family)?;
    let audit:Value=serde_json::from_slice(&acquisition::checked_file(&reg.independent_audit, 4*1024*1024)?)?;
    let run:Value=serde_json::from_slice(&acquisition::checked_file(&reg.run_receipt, 16*1024*1024)?)?;
    ensure!(audit["schema"]=="hf2q.c2.fresh-fixed-input-audit.v1" && audit["verdict"]=="Accepted"
        && audit["capture_sha256"]==reg.capture.sha256 && audit["run_receipt_sha256"]==reg.run_receipt.sha256
        && audit["owner_decision_sha256"]==validated.capture.owner_decision.sha256
        && audit["all_lanes_all_turns"]==true && audit["unforced_native_tapes"]==true
        && audit["historical_archives_unchanged"]==true && audit["real_task_quality_proved"]==false,
        "independent capture audit incomplete/unaccepted");
    super::fresh_admission::validate(&run, &reg.capture, &validated.capture, &validated.inputs.workload_kind)?;
    Ok(validated)
}
pub(crate) fn load_registration(binding:&acquisition::FileBinding) -> Result<Validated> {
    registration_inputs(&serde_json::from_slice(&acquisition::checked_file(binding, 4*1024*1024)?)?)
}
pub(super) fn register() -> Result<()> {
    let path=PathBuf::from(std::env::var_os("HF2Q_C2_FRESH_REGISTER_PACKET").context("fresh registration packet missing")?);
    let packet:RegisterPacket=serde_json::from_slice(&fs::read(path)?)?;
    ensure!(packet.schema=="hf2q.c2.fresh-fixed-input-register-packet.v1", "fresh register packet schema");
    let registration=Registration {schema:REGISTRATION_SCHEMA.into(),reference_id:packet.capture.sha256.clone(),
        capture:packet.capture,independent_audit:packet.independent_audit,run_receipt:packet.run_receipt};
    let validated=registration_inputs(&registration)?;
    output_preflight(&packet.output)?;fs::create_dir(&packet.output)?;
    fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
    fs::write(packet.output.join("registration.json"),serde_json::to_vec_pretty(&registration)?)?;
    let mut differences=Vec::new();
    for (lane, inputs) in validated.inputs.lanes.iter().enumerate() { for (turn, input) in inputs.turns.iter().enumerate() {
        let observed=&validated.observations[lane][turn];
        let old:Value=serde_json::from_slice(&acquisition::checked_file(&input.receipt,4*1024*1024)?)?;
        differences.push(serde_json::json!({"lane":lane,"turn":turn,"request_sha256":input.request.sha256,
            "historical_response_sha256":input.response.sha256,"fresh_response_sha256":observed.response.sha256,
            "historical_usage":old["usage"],"fresh_prompt_tokens":observed.prepared.prompt.len(),
            "fresh_generated_tokens":observed.terminal.generated.len(),"fresh_decision_tokens":observed.terminal.decisions.len(),
            "fresh_native_step":observed.terminal.native_step,"fresh_http_finish":observed.http_finish_reason,
            "historical_native_ids":"UNAVAILABLE: not inferred or recovered","historical_tools_newly_executed":false}));
    }}
    fs::write(packet.output.join("historical-differences.json"),serde_json::to_vec_pretty(&differences)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fresh_inputs_refuse_replacement_original_inventory() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("replacement-inventory.json");
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema":"hf2q.private.c2-original-http-input-inventory.v1",
            "purpose":"historical fixed-input origin only",
            "origins":[{"family":"Qwen36","turn_counts":[7,7,7,7]}]
        })).unwrap();
        fs::write(&path, &bytes).unwrap();
        let substituted = acquisition::FileBinding {path,sha256:hash(&bytes)};
        assert!(original_input_inventory(&substituted).is_err());
    }
    #[test]
    fn fresh_terminal_and_publication_counter_must_reconcile() {
        let mut terminal=Terminal{request_sha256:"a".repeat(64),slot:0,decisions:vec![4,8],generated:vec![4],finish_reason:"stop".into(),native_step:1};
        let stop=StopSelection{token:8,cause:"eos".into()};
        assert!(require_terminal(&terminal,Some(&stop),"generated_tokens",1,"stop").is_ok());
        assert!(require_terminal(&terminal,Some(&stop),"generated_tokens",1,"tool_calls").is_ok());
        assert!(require_terminal(&terminal,Some(&stop),"generated_tokens",1,"length").is_err());
        assert!(require_terminal(&terminal,Some(&stop),"decision_tokens",2,"stop").is_err());
        assert!(require_terminal(&terminal,Some(&stop),"generated_tokens",2,"stop").is_err());
        assert!(require_terminal(&terminal,None,"generated_tokens",1,"stop").is_err());
        terminal.decisions[1]=9;
        assert!(require_terminal(&terminal,Some(&stop),"generated_tokens",1,"stop").is_err());
        terminal.decisions=vec![4];terminal.finish_reason="length".into();
        assert!(require_terminal(&terminal,None,"generated_tokens",1,"length").is_err());
        terminal.generated=vec![4;8192];terminal.decisions=terminal.generated.clone();
        assert!(require_terminal(&terminal,None,"generated_tokens",8192,"length").is_ok());
        assert!(require_terminal(&terminal,None,"generated_tokens",8192,"stop").is_err());
        assert!(require_terminal(&terminal,None,"generated_tokens",8192,"tool_calls").is_err());
        assert!(require_terminal(&terminal,Some(&stop),"generated_tokens",8192,"length").is_err());
    }
    #[test]
    fn fresh_ingress_requires_one_matching_actual_ticket_header() {
        let headers=serde_json::json!({"ticket":"c2-7","headers":[
            ["content-type",[97]],["x-hf2q-c2-ticket",b"c2-7"]]});
        assert!(require_request_ticket(&headers,"c2-7").is_ok());
        let mut missing=headers.clone();missing["headers"]=serde_json::json!([]);
        assert!(require_request_ticket(&missing,"c2-7").is_err());
        let mut duplicate=headers.clone();
        duplicate["headers"].as_array_mut().unwrap().push(serde_json::json!(["X-HF2Q-C2-TICKET",b"c2-7"]));
        assert!(require_request_ticket(&duplicate,"c2-7").is_err());
        let mut different=headers.clone();different["headers"][1][1]=serde_json::json!(b"c2-8");
        assert!(require_request_ticket(&different,"c2-7").is_err());
        let mut nonbytes=headers.clone();nonbytes["headers"][1][1]=serde_json::json!("c2-7");
        assert!(require_request_ticket(&nonbytes,"c2-7").is_err());
        assert!(require_request_ticket(&headers,"c2-8").is_err());
    }
}
