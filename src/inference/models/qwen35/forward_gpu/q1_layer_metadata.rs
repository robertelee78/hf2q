//! Actual loaded handles and host encode-state; no inferred native kernel name.
use super::*;
use serde_json::{json,Value};
fn weight(name:&str,buffer:&MlxBuffer,codec:GgmlType)->Value {
    json!({"name":name,"dtype":format!("{:?}",buffer.dtype()),"codec":format!("{codec:?}"),
        "actual_shape":buffer.shape(),"logical_bytes":buffer.data_byte_len(),
        "native_kernel_route":"UnknownNotObserved"})
}
pub(super) fn ffn(w:&DenseFfnWeightsGpuQ)->Value {
    json!({"gate":weight("gate",&w.gate_q,w.ggml_type_gate_up),
        "up":weight("up",&w.up_q,w.ggml_type_gate_up),
        "down":weight("down",&w.down_q,w.ggml_type_down),
        "gate_up_n":w.intermediate_size,"gate_up_k":w.hidden_size,
        "down_n":w.hidden_size,"down_k":w.intermediate_size})
}
pub(super) fn state(kv:&HybridKvCache,layer:usize,slot:SlotId)->Result<Value> {
    let local=match kv.slot_index_for_layer(layer as u32).context("layer slot missing")? {
        super::super::kv_cache::LayerSlot::Linear(index)=> {
            let s=&kv.linear_attn[index as usize];
            let conv=s.conv_bufs_for_slot(slot).0;let recurrent=s.recurrent_bufs_for_slot(slot).0;
            json!({"kind":"DeltaNet","index":index,"pp_flipped":s.pp_flipped[slot.0 as usize],
                "current_conv_view_offset":conv.byte_offset(),"current_conv_logical_bytes":conv.data_byte_len(),
                "current_recurrent_view_offset":recurrent.byte_offset(),"current_recurrent_logical_bytes":recurrent.data_byte_len()})
        }
        super::super::kv_cache::LayerSlot::Full(index)=>json!({"kind":"FullAttention","index":index,
            "actual_current_len":kv.full_attn[index as usize].current_len[slot.0 as usize]}),
    };
    Ok(json!({"physical_slot":slot.0,"slot_generation":null,"local":local,
        "actual_la_capture_active":kv.la_capture_active(),
        "mtp_cursor":kv.mtp_slot.as_ref().map(|s|s.current_len[slot.0 as usize]),
        "state_read":"host encode-side fields, not GPU Running/completion"}))
}
pub(super) fn layer(weights:&LayerWeightsGpu,kv:&HybridKvCache,layer:usize,slot:SlotId,
    rows:u32,cfg:&super::super::Qwen35Config,before:&Value)->Result<Value> {
    let (kind,projections,ffn_weights)=match weights {
        LayerWeightsGpu::LinearAttn{attn,ffn}=> ("DeltaNet",json!({
            "qkv":weight("qkv",&attn.attn_qkv,attn.attn_qkv_ggml_type),
            "z":weight("z",&attn.attn_gate,attn.attn_gate_ggml_type),
            "alpha":weight("alpha",&attn.ssm_alpha,attn.ssm_alpha_ggml_type),
            "beta":weight("beta",&attn.ssm_beta,attn.ssm_beta_ggml_type),
            "output":weight("output",&attn.ssm_out,attn.ssm_out_ggml_type)}),ffn),
        LayerWeightsGpu::FullAttn{attn,ffn}=> {
            let q=match &attn.q_gate {
                super::super::gpu_full_attn::FullAttnQGateWeightsGpu::Fused{weight:w,ggml_type}=>
                    json!({"storage":"native_fused_q_gate","weight":weight("q_gate",w,*ggml_type)}),
                super::super::gpu_full_attn::FullAttnQGateWeightsGpu::Split{wq,wq_ggml_type,w_gate,w_gate_ggml_type}=>
                    json!({"storage":"split","q":weight("q",wq,*wq_ggml_type),"gate":weight("gate",w_gate,*w_gate_ggml_type)}),
            };
            ("FullAttention",json!({"q_gate":q,"k":weight("k",&attn.wk,attn.wk_ggml_type),
                "v":weight("v",&attn.wv,attn.wv_ggml_type),"output":weight("output",&attn.wo,attn.wo_ggml_type)}),ffn)
        }
    };
    let FfnWeightsGpu::DenseQ(w)=ffn_weights else {anyhow::bail!("selected dense-Q target only")};
    let geometry=if kind=="DeltaNet" {
        let shape=DeltaNetLayerShape::from_config(cfg);
        json!({"h":shape.hidden_size,"n_key_heads":shape.n_k_heads,"n_value_heads":shape.n_v_heads,
            "d_k":shape.d_k,"d_v":shape.d_v,"conv_kernel":shape.conv_kernel,
            "qkv_n":2*shape.n_k_heads*shape.d_k+shape.n_v_heads*shape.d_v,
            "z_n":shape.n_v_heads*shape.d_v,"alpha_beta_n":shape.n_v_heads,
            "projection_input_k":shape.hidden_size,"output_k":shape.n_v_heads*shape.d_v})
    }else{
        let shape=FullAttnShape::from_config(cfg);
        json!({"h":shape.hidden_size,"n_query_heads":shape.n_head,"n_kv_heads":shape.n_kv,
            "head_dim":shape.head_dim,"rotary_dim":shape.rotary_dim,
            "q_gate_fused_n":2*shape.n_head*shape.head_dim,"k_v_n":shape.n_kv*shape.head_dim,
            "projection_input_k":shape.hidden_size,"output_k":shape.n_head*shape.head_dim})
    };
    Ok(json!({"geometry_from_same_actual_cfg":geometry,"attention_kind":kind,"attention_hf_branch":if rows>1 {"arena multirow"}else{"pooled scalar"},
        "actual_m":rows,"projections":projections,"ffn":ffn(w),"before":before,
        "after_attention":state(kv,layer,slot)?,"gdn_chunk_parallel_eligible":false,
        "native_kernel_route":"UnknownNotObserved",
        "native_cached_policy":"UnknownNotObserved","selected_process_env":{"HF2Q_DECODE_MVN":std::env::var("HF2Q_DECODE_MVN").ok(),
            "HF2Q_DECODE_MV_EXT":std::env::var("HF2Q_DECODE_MV_EXT").ok()}}))
}
