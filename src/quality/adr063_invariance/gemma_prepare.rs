//! Metadata-only preparation from retained ACTUAL Rust-tokenized Gemma inputs.
//! No model, context, tensor loader, device, native command or inference.
use super::{acquisition::{self,FileBinding,FileIdentity},protocol::Workload,receipt::Family,store::hash};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use serde_json::Value;
use std::{fs::{File,OpenOptions},io::{Read,Write},os::unix::fs::{OpenOptionsExt,PermissionsExt},path::{Path,PathBuf}};
const MODEL_SHA:&str="bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {path:PathBuf,retained_full_sha256:String,file_identity:FileIdentity}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)] routing_recipes:bool,
    model:Model,full_content_receipt:FileBinding,tokenizer:FileBinding,template:FileBinding,
    producer_metadata:FileBinding,producer_build_receipt:FileBinding,
    producer_export_receipt:FileBinding,producer_config:FileBinding,
    producer_vectors:[FileBinding;4],expected_exporter_source_commit:String,
    expected_exporter_binary_sha256:String,output:PathBuf,
}
fn exact_hex(s:&str,n:usize)->bool {s.len()==n && s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))}
fn output_preflight(path:&Path)->Result<()> {
    ensure!(path.is_absolute() && !path.ancestors().any(|p|p.join(".git").exists()),"private output outsideGit");
    ensure!(path.parent().context("output parent")?.canonicalize()?==path.parent().unwrap(),"canonical outputparent");
    match std::fs::symlink_metadata(path) {Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(()),
        _=>anyhow::bail!("new output only")}
}
fn require_header(architecture:Option<&str>,embedded:Option<&str>,expected:&str)->Result<()> {
    ensure!(architecture==Some("gemma4") && embedded==Some(expected),
        "explicit Gemma architecture and exact embedded template required");Ok(())
}
fn derive(vectors:&[Vec<u32>;4],vocab:u32)->Result<Workload> {
    ensure!(vocab>0 && vectors.iter().all(|v|v.len()==253952 && v.iter().all(|&id|id<vocab)),
        "actual complete selected producer vectors/vocabulary required");
    let work=Workload {family:Family::Gemma4,vocab_size:vocab,
        lanes:vectors.iter().map(|v|v[..144].to_vec()).collect()};
    work.validate().map_err(anyhow::Error::msg)?;Ok(work)
}
fn le_hash(ids:&[u32])->String {use sha2::{Digest,Sha256};let mut h=Sha256::new();
    for id in ids {h.update(id.to_le_bytes());}hex::encode(h.finalize())}
fn planned(cfg:&crate::serve::config::Gemma4Config)->Result<Value> {
    let logit_bytes=4u64.checked_mul(144).and_then(|n|n.checked_mul(cfg.vocab_size as u64))
        .and_then(|n|n.checked_mul(4)).context("Gemma logits overflow")?;
    let expected=super::gemma_cache::expected(cfg,&[0,1,2,3],&[(64,false),(64,true),(128,false),(144,false)])
        .map_err(anyhow::Error::msg)?;
    let cache_bytes=expected.iter().try_fold(0u64,|sum,e|e.shape.iter()
        .try_fold(e.scalar.bytes() as u64,|n,&d|n.checked_mul(d as u64)).and_then(|n|sum.checked_add(n)))
        .context("Gemma cache capture overflow")?;
    let total=logit_bytes.checked_add(cache_bytes).context("Gemma total capture overflow")?;
    ensure!(total<=2*1024*1024*1024,"actual endpoint payload exceeds existing2GiB ceiling");
    let anchor= (0..cfg.num_hidden_layers).filter(|&l|!cfg.is_full_attention(l)).try_fold(0u64,|sum,l| {
        let hd=cfg.head_dim_for_layer(l) as u64;let heads=cfg.num_kv_heads_for_layer(l) as u64;
        heads.checked_mul(cfg.sliding_window as u64).and_then(|n|n.checked_mul(3*hd+4*(hd/256)))
            .and_then(|n|sum.checked_add(n))
    }).context("Gemma anchor overflow")?;
    Ok(serde_json::json!({"max_endpoint_payload":total,"full144_logits":logit_bytes,
        "active_cache_with_exact64_pair":cache_bytes,"one_live_serial_anchor_payload":anchor,
        "metadata_store_ceiling":134217728,"physical_admission":false,
        "operating_budget_disposition":"root must charge owned I/O/copies/model/fullC4scaffolds and positive margins separately"}))
}
fn routing_recipe_export(cfg:&crate::serve::config::Gemma4Config)->Result<Vec<Value>> {
    let mut rows=Vec::new();
    for routing in [false,true] {
        for (adapter,recipe) in [
            ("ordinary",super::gemma_actual::selected_recipe(cfg,routing)),
            ("T5",super::gemma_ngram::configuration_recipe(cfg,routing)),
        ] {
            let digest=acquisition::config_digest(&recipe)?;
            rows.push(serde_json::json!({"adapter":adapter,"routing_witness":routing,
                "config_sha256":digest,"recipe":recipe}));
        }
    }
    Ok(rows)
}
#[test]
#[ignore="root-supervised header/retainedJSON only; no model/device"]
fn q1_gemma_endpoint_metadata_prepare(){run().unwrap();}
fn run()->Result<()> {
    let path=std::env::var_os("HF2Q_Q1_GEMMA_PREP_CONFIG").context("explicit Gemma prep config")?;
    let mut raw=Vec::new();File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=1024*1024,"bounded metadata config");let c:Config=serde_json::from_slice(&raw)?;
    output_preflight(&c.output)?;
    if c.routing_recipes {ensure!(env!("HF2Q_MLX_NATIVE_VERSION")==super::gemma_routing_witness::WITNESS_NATIVE_VERSION,"routing recipe metadata requires the exact published routing-witness native build");}
    ensure!(exact_hex(&c.expected_exporter_source_commit,40)&&exact_hex(&c.expected_exporter_binary_sha256,64)
        &&c.model.retained_full_sha256==MODEL_SHA,"exact selected source/binary/model bindings");
    let source=option_env!("HF2Q_BUILD_GIT_SHA").context("compiled source stamp")?;
    ensure!(source==c.expected_exporter_source_commit,"actual exporter source mismatch");
    let binary=acquisition::executable_digest()?;ensure!(binary==c.expected_exporter_binary_sha256,"actual executable mismatch");
    let bindings=[(&c.full_content_receipt,16*1024*1024),(&c.tokenizer,64*1024*1024),
        (&c.template,1024*1024),(&c.producer_metadata,2*1024*1024),(&c.producer_build_receipt,2*1024*1024),
        (&c.producer_export_receipt,2*1024*1024),(&c.producer_config,1024*1024)];
    let bound=bindings.iter().map(|(b,n)|acquisition::checked_file(b,*n)).collect::<Result<Vec<_>>>()?;
    acquisition::full_content_receipt_has_model(&bound[0],MODEL_SHA)?;
    std::str::from_utf8(&bound[1])?;let template=std::str::from_utf8(&bound[2])?;
    let metadata:Value=serde_json::from_slice(&bound[3])?;let build:Value=serde_json::from_slice(&bound[4])?;
    let export:Value=serde_json::from_slice(&bound[5])?;let producer_config:Value=serde_json::from_slice(&bound[6])?;
    ensure!(metadata["schema"]=="hf2q.gemma-largest-prompt-metadata.v1" && metadata["family"]=="gemma4"
        &&metadata["prompt_tokens"]==253952 &&metadata["context"]==262144
        &&metadata["model_inference_performed"]==false &&metadata["config_sha256"]==c.producer_config.sha256
        &&metadata["tokenizer_raw_sha256"]==c.tokenizer.sha256
        &&metadata["template_raw_sha256"]==c.template.sha256,"retained actual Gemma metadata bindings");
    ensure!(build["source_clean"]==true &&build["model_loaded"]==false
        &&build["source_commit"]==build["explicit_git_commit_sha"] &&build["source_commit"]==metadata["compiled_build_git_sha"]
        &&export["helper_source"]==build["source_commit"] &&export["actual_metadata_export_passed"]==true
        &&export["model_loaded"]==false &&export["metadata_sha256"]==c.producer_metadata.sha256,
        "retained metadata build/export source join; not new execution approval");
    ensure!(producer_config["family"]=="gemma4" &&producer_config["tokenizer"]["sha256"]==c.tokenizer.sha256
        &&producer_config["template"]["sha256"]==c.template.sha256,"producer tokenizer/template config join");
    let mut vectors:[Vec<u32>;4]=std::array::from_fn(|_|Vec::new());let mut vector_raw=Vec::new();
    let requests=metadata["requests"].as_array().context("producer four requests")?;ensure!(requests.len()==4,"four producer vectors");
    for lane in 0..4 {
        let b=&c.producer_vectors[lane];let name=format!("agent-{lane}-token-ids.json");
        ensure!(b.path.parent()==c.producer_metadata.path.parent() &&b.path.file_name().and_then(|s|s.to_str())==Some(name.as_str()),"actual ordered producer filenames");
        let bytes=acquisition::checked_file(b,4*1024*1024)?;vectors[lane]=serde_json::from_slice(&bytes)?;
        ensure!(requests[lane]["agent"]==lane &&requests[lane]["token_file_sha256"]==b.sha256
            &&requests[lane]["token_ids_sha256_le_u32"]==le_hash(&vectors[lane])
            &&export["artifacts"][name.as_str()]==b.sha256,"actual raw/LE producer vector identity");
        vector_raw.push(bytes);
    }
    ensure!(c.model.path.is_absolute(),"absolute model path");let fd=File::open(&c.model.path)?;
    let before=FileIdentity::observed(&fd)?;ensure!(before==c.model.file_identity,"actual model descriptor/stat mismatch");
    let gguf=mlx_native::gguf::GgufFile::from_file(fd.try_clone()?)?;
    ensure!(gguf.tensor_data_offset()<=32*1024*1024,"bounded header only");
    require_header(gguf.metadata_string("general.architecture"),gguf.metadata_string("tokenizer.chat_template"),template)?;
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&gguf)?;cfg.max_position_embeddings=262144;
    let work=derive(&vectors,u32::try_from(cfg.vocab_size)?)?;
    let recipe=super::gemma_actual::recipe(&cfg);let plan=planned(&cfg)?;
    let workload_bytes=serde_json::to_vec_pretty(&work)?;
    let value=serde_json::json!({"schema":"hf2q.q1.gemma-endpoint-preparation.v1","exporter_source_commit":source,
        "exporter_binary_sha256":binary,"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "family":Family::Gemma4,"configured_context":262144,"output_allowance":8192,"logical_lanes":4,"tokens_per_lane":144,
        "config_input_sha256":hash(&raw),"model_path":c.model.path,"workload_file_sha256":hash(&workload_bytes),
        "full_content_receipt_path":c.full_content_receipt.path,"full_content_receipt_sha256":c.full_content_receipt.sha256,
        "model_full_sha256_retained":MODEL_SHA,"model_file_identity":before,
        "model_payload_rehashed":false,"header_data_offset":gguf.tensor_data_offset(),
        "tokenizer_file_sha256":c.tokenizer.sha256,"template_file_sha256":c.template.sha256,
        "producer_metadata_sha256":c.producer_metadata.sha256,"producer_vector_sha256":c.producer_vectors.iter().map(|b|&b.sha256).collect::<Vec<_>>(),
        "token_vectors_sha256":acquisition::tokens_digest(&work)?,"actual_gemma_config_debug":format!("{cfg:?}"),
        "actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,"config_sha256":acquisition::config_digest(&recipe)?,
        "recipe":recipe,"payload_plan":plan,"derivation":"first144 retained actual Rust input IDs:128prompt+16forced; not generated canonical continuation",
        "native_checksum_verification":"independent published build composition required","model_loaded":false,
        "device_created":false,"inference_performed":false,"tokenizer_semantic_attestation":false,
        "execution_approved":false,"composition_reviewed":false,"acquisition_attested":false,"whole_q1_qualified":false});
    let encoded=serde_json::to_vec_pretty(&value)?;ensure!(encoded.len()<=2*1024*1024,"metadata output bound");
    // Additive file only; omitted/false flag preserves both old output bytes.
    let routing_encoded=if c.routing_recipes {
        let recipes=routing_recipe_export(&cfg)?;
        let v=serde_json::json!({"schema":"hf2q.q1.gemma-routing-recipes-metadata.v1",
            "exporter_source_commit":source,"exporter_binary_sha256":binary,
            "native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"config_input_sha256":hash(&raw),
            "model_full_sha256_retained":MODEL_SHA,"model_file_identity":before,
            "header_data_offset":gguf.tensor_data_offset(),"actual_gemma_config_debug":format!("{cfg:?}"),
            "actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,
            "tokenizer_file_sha256":c.tokenizer.sha256,"template_file_sha256":c.template.sha256,
            "recipes":recipes,"runtime_dataset_binding":"SEPARATE independently reviewed842 greedy144 dataset; no dataset identity attested by recipe export",
            "model_loaded":false,"device_created":false,"inference_performed":false,
            "acquisition_attested":false,"execution_approved":false,"physical_admission":false,"whole_q1_qualified":false});
        let encoded=serde_json::to_vec_pretty(&v)?;ensure!(encoded.len()<=128*1024,"bounded recipe metadata");Some(encoded)
    }else{None};
    for ((b,n),old) in bindings.iter().zip(&bound) {ensure!(acquisition::checked_file(b,*n)?==*old,"bound input changed");}
    for (b,old) in c.producer_vectors.iter().zip(&vector_raw) {ensure!(acquisition::checked_file(b,4*1024*1024)?==*old,"producer vector changed");}
    ensure!(FileIdentity::observed(&fd)?==before,"model descriptor changed during metadata read");
    output_preflight(&c.output)?;std::fs::create_dir(&c.output)?;std::fs::set_permissions(&c.output,std::fs::Permissions::from_mode(0o700))?;
    for (name,bytes) in [("prepared.json",encoded),("workload.json",workload_bytes)] {
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join(name))?.write_all(&bytes)?;
    }
    if let Some(bytes)=routing_encoded {
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join("routing-recipes.json"))?.write_all(&bytes)?;
        println!("Q1_GEMMA_ROUTING_RECIPES_METADATA_PREPARED recipes=4 dataset_attested=false model=false device=false execution=false");
    }
    println!("Q1_GEMMA_ENDPOINT_METADATA_PREPARED lanes=4 tokens=144 device=false execution=false");Ok(())
}
#[cfg(test)]
#[path="gemma_prepare_controls.rs"]
mod controls;
