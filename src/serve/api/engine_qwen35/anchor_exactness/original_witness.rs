//! Immutable original-W native token/terminal evidence. A replacement HTTP
//! replay is not authority for W, even when its usage counts happen to agree.
use super::*;
use super::frozen_script::{Prepared, Terminal};
use serde_json::Value;

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all="snake_case")]
pub(crate) enum CompletionCounter { GeneratedTokens, DecisionTokens, NativeStep }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OriginalWitness {
    pub(crate) schema:String,
    pub(crate) request_sha256:String,
    pub(crate) response_sha256:String,
    pub(crate) producer_identity:Identity,
    pub(crate) prompt_tokens:Vec<u32>,
    pub(crate) generated_tokens:Vec<u32>,
    pub(crate) decision_tokens:Vec<u32>,
    pub(crate) native_step:usize,
    pub(crate) native_finish_reason:String,
    pub(crate) http_finish_reason:String,
    pub(crate) terminal_token:Option<u32>,
    pub(crate) usage_completion_counter:CompletionCounter,
}
pub(crate) struct WireTerminal {
    pub(crate) prompt_tokens:usize,
    pub(crate) completion_tokens:usize,
    pub(crate) finish_reason:String,
}
impl OriginalWitness {
    pub(crate) fn load(binding:&acquisition::FileBinding,request_sha:&str,response_sha:&str,
        family:Family,stream:bool,receipt:&Value,response:&[u8])->Result<Self> {
        let raw=acquisition::checked_file(binding,64*1024*1024)
            .context("AcquisitionBlocked: original continuation/terminal witness unavailable or changed")?;
        let witness:Self=serde_json::from_slice(&raw)
            .context("AcquisitionBlocked: original continuation/terminal witness invalid")?;
        witness.producer_identity.validate_c2_historical().map_err(anyhow::Error::msg)?;
        ensure!(witness.schema=="hf2q.c2.original-continuation-terminal-witness.v1"
            &&witness.request_sha256==request_sha&&witness.response_sha256==response_sha,
            "AcquisitionBlocked: original witness does not bind original raw request/response bytes");
        ensure!(witness.producer_identity.family==family
            &&witness.producer_identity.model_sha256==match family {Family::Qwen38=>Q38,Family::Qwen36=>Q36,_=>anyhow::bail!("original W only admits Qwen artifacts")},
            "AcquisitionBlocked: original witness producer artifact differs");
        // These are the immutable ORIGINAL W producer pins, not the current
        // diagnostic replay's source/native identity. A fresh replay may not
        // relabel its observation as a missing historical witness.
        ensure!(witness.producer_identity.source_commit=="51a3923fafbe46e4a18c22c5fdc24ce20a46593c"
            &&witness.producer_identity.native_version=="0.18.0"
            &&witness.producer_identity.native_checksum=="2a804cf31840059e548c763be935be6b6b75563add645556311130a61ed525fd",
            "AcquisitionBlocked: original witness producer source/native differs from frozen W");
        ensure!(!witness.prompt_tokens.is_empty()&&witness.prompt_tokens.len()+8192<=262144
            &&!witness.generated_tokens.is_empty()&&witness.generated_tokens.len()<=8192
            &&witness.decision_tokens.starts_with(&witness.generated_tokens)
            &&witness.decision_tokens.len()>=witness.generated_tokens.len()
            &&witness.decision_tokens.len()<=witness.generated_tokens.len()+1
            &&witness.terminal_token==witness.decision_tokens.last().copied()
            &&matches!(witness.native_finish_reason.as_str(),"stop"|"length")
            &&matches!(witness.http_finish_reason.as_str(),"stop"|"tool_calls"|"length"),
            "AcquisitionBlocked: original native token/terminal witness incomplete");
        let wire=wire_terminal(response,stream)?;
        let counter=match witness.usage_completion_counter {
            CompletionCounter::GeneratedTokens=>witness.generated_tokens.len(),
            CompletionCounter::DecisionTokens=>witness.decision_tokens.len(),
            CompletionCounter::NativeStep=>witness.native_step,
        };
        ensure!(wire.prompt_tokens==witness.prompt_tokens.len()&&wire.completion_tokens==counter
            &&receipt["usage"]["prompt_tokens"].as_u64()==Some(wire.prompt_tokens as u64)
            &&receipt["usage"]["completion_tokens"].as_u64()==Some(wire.completion_tokens as u64)
            &&wire.finish_reason==witness.http_finish_reason,
            "AcquisitionBlocked: original usage/finish cannot be reconciled with its explicitly witnessed native counter; no trimming/padding");
        Ok(witness)
    }
    pub(crate) fn require_native(&self,prepared:&Prepared,terminal:&Terminal)->Result<()> {
        ensure!(prepared.prompt==self.prompt_tokens,
            "AcquisitionBlocked: actual prepared prompt differs from original token witness");
        require_original_sequence(&self.generated_tokens,&self.decision_tokens,
            &self.native_finish_reason,self.terminal_token,terminal)
    }
    pub(crate) fn require_response(&self,bytes:&[u8],stream:bool,prepared:&Prepared,terminal:&Terminal)->Result<()> {
        self.require_native(prepared,terminal)?;
        let wire=wire_terminal(bytes,stream)?;
        // Actual current production Qwen35DecodeState::finish publishes
        // generated_tokens.len() as GenerationResult.completion_tokens. Keep
        // the original producer's declared counter above separate and intact.
        ensure!(wire.prompt_tokens==prepared.prompt.len()
            &&wire.completion_tokens==terminal.generated.len(),
            "AcquisitionBlocked: current actual HTTP usage disagrees with its verbatim native token tape");
        require_http_boundary(&self.http_finish_reason,&wire.finish_reason)
    }
}
fn require_original_sequence(generated:&[u32],decisions:&[u32],finish:&str,terminal_token:Option<u32>,
    actual:&Terminal)->Result<()> {
    ensure!(actual.generated==generated&&actual.decisions==decisions
        &&actual.finish_reason==finish&&actual.decisions.last().copied()==terminal_token,
        "AcquisitionBlocked: replay changed original continuation or terminal token; replacement output cannot define W");
    Ok(())
}
fn require_http_boundary(original:&str,actual:&str)->Result<()> {
    ensure!(original==actual,"AcquisitionBlocked: actual tool_calls/stop/length boundary differs from original witness");
    Ok(())
}
pub(crate) fn wire_terminal(bytes:&[u8],stream:bool)->Result<WireTerminal> {
    let mut usage=None;let mut finish=None;
    if stream {
        let text=std::str::from_utf8(bytes)?;let mut done=false;
        for line in text.lines() {
            let Some(data)=line.strip_prefix("data: ") else{continue};
            ensure!(!done,"HTTP data follows SSE terminal");
            if data=="[DONE]" {done=true;continue}
            let chunk:Value=serde_json::from_str(data)?;
            ensure!(chunk.get("error").is_none(),"HTTP error cannot attest original token extents");
            if chunk.get("usage").is_some_and(|v|!v.is_null()) {
                ensure!(usage.is_none(),"duplicate HTTP usage");usage=Some(chunk["usage"].clone());
            }
            if let Some(reason)=chunk["choices"][0]["finish_reason"].as_str() {
                ensure!(finish.is_none(),"duplicate HTTP finish");finish=Some(reason.to_owned());
            }
        }
        ensure!(done,"actual SSE terminal missing");
    }else{
        let response:Value=serde_json::from_slice(bytes)?;
        ensure!(response.get("error").is_none(),"HTTP error cannot attest original W");
        usage=Some(response["usage"].clone());
        finish=response["choices"][0]["finish_reason"].as_str().map(str::to_owned);
    }
    let usage=usage.context("actual HTTP usage missing")?;
    let finish=finish.context("actual HTTP finish missing")?;
    ensure!(matches!(finish.as_str(),"stop"|"tool_calls"|"length"),"incomplete HTTP finish");
    Ok(WireTerminal {prompt_tokens:usize::try_from(usage["prompt_tokens"].as_u64().context("prompt usage missing")?)?,
        completion_tokens:usize::try_from(usage["completion_tokens"].as_u64().context("completion usage missing")?)?,finish_reason:finish})
}

#[test]
fn c2_original_witness_rejects_equal_count_replacement_and_terminal_changes() {
    let mut actual=Terminal {request_sha256:"a".repeat(64),slot:0,generated:vec![17,29],
        decisions:vec![17,29,151645],finish_reason:"stop".into(),native_step:3};
    assert!(require_original_sequence(&[17,29],&[17,29,151645],"stop",Some(151645),&actual).is_ok());
    actual.generated[1]=31;actual.decisions[1]=31;
    assert!(require_original_sequence(&[17,29],&[17,29,151645],"stop",Some(151645),&actual).is_err());
    actual.generated[1]=29;actual.decisions[1]=29;actual.decisions[2]=151643;
    assert!(require_original_sequence(&[17,29],&[17,29,151645],"stop",Some(151645),&actual).is_err());
    assert!(require_http_boundary("tool_calls","stop").is_err());
}

#[test]
fn historical_c2_native_identity_is_not_current_runtime_identity() {
    let mut identity=Identity{family:Family::Qwen38,
        source_commit:"51a3923fafbe46e4a18c22c5fdc24ce20a46593c".into(),binary_sha256:"a".repeat(64),
        native_version:"0.18.0".into(),native_checksum:"2a804cf31840059e548c763be935be6b6b75563add645556311130a61ed525fd".into(),
        model_sha256:Q38.into(),tokenizer_sha256:"b".repeat(64),template_sha256:"c".repeat(64),
        config_sha256:"d".repeat(64),token_vectors_sha256:"e".repeat(64)};
    assert!(identity.validate_c2_historical().is_ok());
    assert!(identity.validate().is_err());
    identity.native_version=env!("HF2Q_MLX_NATIVE_VERSION").into();
    assert!(identity.validate_c2_historical().is_err());
    identity.native_version="0.18.0".into();identity.native_checksum="f".repeat(64);
    assert!(identity.validate_c2_historical().is_err());
}
