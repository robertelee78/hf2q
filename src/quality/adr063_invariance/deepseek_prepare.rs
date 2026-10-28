//! Metadata-only native DS replay/header preparation, never model/device load.
use super::{acquisition::{self,FileBinding,FileIdentity},protocol::{Trajectory,Workload},receipt::Family,store::hash};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{fs::{File,OpenOptions},io::{Read,Write},os::unix::fs::{OpenOptionsExt,PermissionsExt},path::{Path,PathBuf}};
const MODEL_SHA:&str="50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {path:PathBuf,retained_full_sha256:String,file_identity:FileIdentity}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Producer {config:FileBinding,proof:FileBinding,request:FileBinding}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    model:Model,full_content_receipt:FileBinding,tokenizer:FileBinding,template:FileBinding,
    producer_execution:FileBinding,producer_helper_binding:FileBinding,producers:[Producer;4],
    expected_exporter_source_commit:Option<String>,expected_exporter_binary_sha256:Option<String>,output:PathBuf,
}
fn hex(s:&str,n:usize)->bool{s.len()==n&&s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))}
fn preflight(path:&Path)->Result<()> {
    ensure!(path.is_absolute()&&!path.ancestors().any(|p|p.join(".git").exists()),"private output outsideGit");
    ensure!(path.parent().context("parent")?.canonicalize()?==path.parent().unwrap(),"canonical outputparent");
    ensure!(matches!(std::fs::symlink_metadata(path),Err(e) if e.kind()==std::io::ErrorKind::NotFound),"new output only");Ok(())
}
fn derive(ids:&[Vec<u32>;4],vocab:u32)->Result<Workload>{
    ensure!(ids.iter().all(|p|p.len()>=144&&p.len()<=253952&&p.iter().all(|&x|x<vocab)),"complete real DS vectors");
    let w=Workload{family:Family::Deepseek4,vocab_size:vocab,lanes:ids.iter().map(|p|p[..144].to_vec()).collect()};
    w.validate().map_err(anyhow::Error::msg)?;Ok(w)
}
fn le_hash(ids:&[u32])->String{use sha2::{Digest,Sha256};let mut h=Sha256::new();for id in ids{h.update(id.to_le_bytes());}hex::encode(h.finalize())}
fn replay_matches(ids:&[u32],rendered:&str,wire_sha:&str,obs:&Value)->Result<()> {
    ensure!(obs["turn"]==0&&obs["request_sha256"]==wire_sha&&obs["prompt_tokens"]==ids.len()
        &&obs["rendered_sha256"]==hash(rendered.as_bytes())
        &&obs["prompt_token_ids_sha256"]==hash(&serde_json::to_vec(ids)?),"retained full native render/token identity BEFORE slice");Ok(())
}
#[test]
#[ignore="root-supervised retainedJSON+sameFD header only; no model/device"]
fn q1_deepseek_endpoint_metadata_prepare(){run().unwrap();}
fn run()->Result<()> {
    let path=std::env::var_os("HF2Q_Q1_DS_PREP_CONFIG").context("explicit DS prep config")?;
    let mut raw=Vec::new();File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=1024*1024,"bounded config");let c:Config=serde_json::from_slice(&raw)?;preflight(&c.output)?;
    let source=c.expected_exporter_source_commit.as_deref().context("pending source binding")?;
    let binary=c.expected_exporter_binary_sha256.as_deref().context("pending binary binding")?;
    ensure!(hex(source,40)&&hex(binary,64)&&option_env!("HF2Q_BUILD_GIT_SHA")==Some(source)
        &&acquisition::executable_digest()?==binary&&c.model.retained_full_sha256==MODEL_SHA,"actual prep source/binary/selected DS identity");
    let mut retained:Vec<(&FileBinding,usize,Vec<u8>)>=Vec::new();
    for (b,cap) in [(&c.full_content_receipt,16*1024*1024),(&c.tokenizer,64*1024*1024),
        (&c.template,1024*1024),(&c.producer_execution,2*1024*1024),(&c.producer_helper_binding,2*1024*1024)] {
        retained.push((b,cap,acquisition::checked_file(b,cap)?));
    }
    acquisition::full_content_receipt_has_model(&retained[0].2,MODEL_SHA)?;
    let tokenizer=tokenizers::Tokenizer::from_bytes(&retained[1].2).map_err(|e|anyhow::anyhow!("tokenizer: {e}"))?;
    ensure!(tokenizer.get_truncation().is_none(),"no tokenizer truncation");
    let template=std::str::from_utf8(&retained[2].2)?.to_owned();
    ensure!(template==crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731,"exact DS NativeEncoding template");
    let execution:Value=serde_json::from_slice(&retained[3].2)?;
    let helper:Value=serde_json::from_slice(&retained[4].2)?;
    let cases=execution["cases"].as_array().context("actual producer executions")?;
    ensure!(cases.len()==4&&helper["family"]=="deepseek4"&&helper["execution_while_loaded"]==false,"actual post-unload DS producer");
    let mut ids:[Vec<u32>;4]=std::array::from_fn(|_|Vec::new());let mut full_vectors=Vec::new();
    let mut observations=Vec::new();
    for lane in 0..4 {
        let p=&c.producers[lane];let config_bytes=acquisition::checked_file(&p.config,1024*1024)?;
        let proof_bytes=acquisition::checked_file(&p.proof,2*1024*1024)?;let wire=acquisition::checked_file(&p.request,1024*1024)?;
        let config:Value=serde_json::from_slice(&config_bytes)?;let proof:Value=serde_json::from_slice(&proof_bytes)?;
        ensure!(cases[lane]["agent"]==lane&&cases[lane]["exit"]==0&&cases[lane]["reaped"]==true
            &&cases[lane]["group_empty"]==true&&cases[lane]["config_sha256"]==p.config.sha256,"actual ordered producer exit/config join");
        ensure!(config["family"]=="deepseek4"&&config["server_slots"]==4
            &&config["tokenizer"]["sha256"]==c.tokenizer.sha256&&config["template"]["sha256"]==c.template.sha256
            &&config["template_source"]=="NativeEncoding { name: \"DEEPSEEK_V4_FLASH_0731\" }"
            &&config["turns"][0]["request"]["sha256"]==p.request.sha256
            &&config["turns"][0]["request"]["path"]==p.request.path.to_str().context("UTF8 request path")?,"actual DS config/native source/request join");
        ensure!(proof["schema"]=="hf2q.deepseek4-prefix-post-oracle.v1"&&proof["family"]=="deepseek4"
            &&proof["config_sha256"]==p.config.sha256&&proof["tokenizer_sha256"]==c.tokenizer.sha256
            &&proof["full_prefix_reuse"]=="PassedExactNativeSevenWireReplay"&&proof["model_inference_performed"]==false
            &&proof["expected_server_source_commit"]==helper["expected_server_source_commit"]
            &&proof["expected_server_binary_sha256"]==helper["expected_server_binary_sha256"],"retained actual native proof provenance");
        let (tokens,rendered)=crate::serve::api::handlers::q1_deepseek_prepare::replay(&wire,&tokenizer,&template)?;
        let obs=&proof["observations"][0];replay_matches(&tokens,&rendered,&p.request.sha256,obs)?;
        observations.push(json!({"agent":lane,"full_prompt_tokens":tokens.len(),"full_prompt_raw_json_sha256":hash(&serde_json::to_vec(&tokens)?),
            "full_prompt_le_u32_sha256":le_hash(&tokens),"rendered_sha256":hash(rendered.as_bytes()),"producer_proof_sha256":p.proof.sha256,"request_sha256":p.request.sha256}));
        full_vectors.push(serde_json::to_vec(&tokens)?);ids[lane]=tokens;
        retained.push((&p.config,1024*1024,config_bytes));retained.push((&p.proof,2*1024*1024,proof_bytes));retained.push((&p.request,1024*1024,wire));
    }
    ensure!(c.model.path.is_absolute(),"absolute selected model");let fd=File::open(&c.model.path)?;
    let before=FileIdentity::observed(&fd)?;ensure!(before==c.model.file_identity,"current exact descriptor/stat");
    let gguf=mlx_native::gguf::GgufFile::from_file(fd.try_clone()?)?;
    ensure!(gguf.tensor_data_offset()<=32*1024*1024&&gguf.metadata_string("general.architecture")==Some("deepseek4")
        &&gguf.metadata_string("tokenizer.chat_template")==Some(template.as_str()),"bounded actual DS header/native template");
    let cfg=crate::inference::models::deepseek4::Deepseek4Config::from_gguf(&gguf)?;
    let workload=derive(&ids,cfg.vocab_size)?;let recipe=super::deepseek_actual::recipe(&cfg);
    let expected=super::deepseek_actual::expected(&cfg,Trajectory::T1Resume64)?;
    let payload=super::deepseek_actual::payload(&expected)?;ensure!(payload<=2*1024*1024*1024,"DS endpoint payload exceeds selected2GiB guard");
    let cache=crate::inference::models::deepseek4::cache::Deepseek4CachePlan::for_context(&cfg,262144)?;
    let workload_bytes=serde_json::to_vec_pretty(&workload)?;
    let value=json!({"schema":"hf2q.q1.deepseek-endpoint-preparation.v1","family":Family::Deepseek4,
        "exporter_source_commit":source,"exporter_binary_sha256":binary,"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "configured_context":262144,"output_allowance":8192,"logical_lanes":4,"physical_cache_owners_planned":4,
        "simultaneously_executing_lanes":1,"ordinary_ownership":"four retained caches, serial logical trajectories; not physical B4",
        "tokens_per_lane":144,"token_vectors_sha256":acquisition::tokens_digest(&workload)?,"workload_file_sha256":hash(&workload_bytes),
        "config_input_sha256":hash(&raw),"model_path":c.model.path,"model_file_identity":before,
        "model_full_sha256_retained":MODEL_SHA,"model_payload_rehashed":false,"header_data_offset":gguf.tensor_data_offset(),
        "full_content_receipt_path":c.full_content_receipt.path,"full_content_receipt_sha256":c.full_content_receipt.sha256,
        "tokenizer_file_sha256":c.tokenizer.sha256,"template_file_sha256":c.template.sha256,"template_source":"NativeEncoding { name: \"DEEPSEEK_V4_FLASH_0731\" }",
        "actual_deepseek_config_debug":format!("{cfg:?}"),"actual_deepseek_config_debug_sha256":recipe.actual_model_config_sha256,
        "config_sha256":acquisition::config_digest(&recipe)?,"recipe":recipe,
        "actual_cache_plan_debug":format!("{cache:?}"),"actual_cache_plan_debug_sha256":hash(format!("{cache:?}").as_bytes()),"producer_replays":observations,
        "payload_plan":{"max_endpoint_payload":payload,"four_fullC_cache_requested_bytes":cache.resident_bytes.checked_mul(4).context("four cache overflow")?,
            "metadata_store_ceiling":134217728,"physical_admission":false,"operating_budget_disposition":"125.6GB operating ledger remains unadmitted; no memory fit certificate"},
        "derivation":"first144 of four full actual Rust nativeDS request replays;128prompt+16forced diagnostics, not generated canonical quality",
        "model_loaded":false,"device_created":false,"inference_performed":false,"execution_approved":false,"composition_reviewed":false,
        "acquisition_attested":false,"tokenizer_semantic_attestation":false,"whole_q1_qualified":false});
    let encoded=serde_json::to_vec_pretty(&value)?;ensure!(encoded.len()<=2*1024*1024,"metadata output bound");
    for (b,cap,old) in &retained {ensure!(acquisition::checked_file(b,*cap)?==*old,"retained input changed");}
    ensure!(FileIdentity::observed(&fd)?==before,"model descriptor changed");preflight(&c.output)?;
    std::fs::create_dir(&c.output)?;std::fs::set_permissions(&c.output,std::fs::Permissions::from_mode(0o700))?;
    let mut files=vec![("prepared.json".to_owned(),encoded),("workload.json".to_owned(),workload_bytes)];
    for (lane,bytes) in full_vectors.into_iter().enumerate(){files.push((format!("agent-{lane}-full-token-ids.json"),bytes));}
    for (name,bytes) in files {OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join(name))?.write_all(&bytes)?;}
    println!("Q1_DEEPSEEK_ENDPOINT_METADATA_PREPARED lanes=4 tokens=144 device=false execution=false");Ok(())
}
#[path="deepseek_prepare_controls.rs"]
mod controls;
