use anyhow::{bail, ensure, Context, Result};
use mlx_native::gguf::{GgufFile, GgufMappedTensorSet};
use mlx_native::{MlxBuffer, MlxDevice};

use super::ffn::MoeFfnShape;
use super::gpu_ffn::{DenseFfnWeightsGpu, DenseFfnWeightsGpuQ, MoeFfnWeightsGpuQ};
use super::gpu_full_attn::{upload_f32_weight, FullAttnQGateWeightsGpu};
use super::mtp::{MtpFfnWeightsGpu, MtpFullAttnWeightsGpu, MtpQGateWeightsGpu, MtpWeights};
use super::weight_loader::{
    dense_ffn_storage, dense_ffn_tensor_types, load_dense_ffn_native, load_dense_ffn_quantized,
    load_f32_tensor, load_moe_ffn_quantized, load_native_projection, DenseFfnStorage,
};
use super::Qwen35Config;
use crate::serve::forward_mlx_shared::MlxQWeight;

pub fn load_mtp_weights_if_present(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<Option<MtpWeights>> {
    let mapped = gguf
        .map_tensor_data(device)
        .context("map GGUF tensor data for standalone MTP load")?;
    load_mtp_weights_if_present_with_shared_head(gguf, &mapped, cfg, device, None)
}

pub fn load_mtp_weights_if_present_with_shared_head(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    device: &MlxDevice,
    main_output_head: Option<&MlxQWeight>,
) -> Result<Option<MtpWeights>> {
    if cfg.mtp_num_hidden_layers == 0 {
        return Ok(None);
    }
    if cfg.mtp_num_hidden_layers != 1 {
        bail!(
            "qwen35 MTP loader supports exactly one nextn layer, got {}",
            cfg.mtp_num_hidden_layers
        );
    }

    let layer_index = cfg.num_hidden_layers;
    let loaded_tensor_names = mtp_tensor_names(gguf, layer_index);
    if loaded_tensor_names.is_empty() {
        bail!(
            "qwen35 metadata advertises nextn_predict_layers=1 but no blk.{layer_index}.nextn.* or blk.{layer_index}.* MTP tensors were found"
        );
    }

    let h = cfg.hidden_size as usize;
    let p = format!("blk.{layer_index}");
    let nextn = format!("{p}.nextn");
    let enorm = load_norm_gpu(gguf, &format!("{nextn}.enorm.weight"), h, device)?;
    let hnorm = load_norm_gpu(gguf, &format!("{nextn}.hnorm.weight"), h, device)?;
    // ADR-013 P14 follow-up (2026-04-30): honor mtp_use_dedicated_embeddings.
    //
    // - True  → MTP carries its own embed table at `blk.{N}.nextn.embed_tokens.weight`
    //           (Qwen3.5 MTP convention).
    // - False → MTP shares the main model's `token_embd.weight` (Qwen3.6 27B + 35B-A3B
    //           convention; convert correctly skips emitting the redundant tensor).
    //           We do NOT duplicate the buffer: `forward_draft` is called with the
    //           per-token embedding already materialised by the verifier's hot path,
    //           and the field itself is reserved for future direct lookups (use the
    //           main model's token_embd via Qwen35Model accessors).
    //
    // Logged at INFO level so operators can confirm the path.
    let embed_tokens_tname = format!("{nextn}.embed_tokens.weight");
    let embed_tokens = if cfg.mtp_use_dedicated_embeddings {
        let info = gguf.tensor_info(&embed_tokens_tname).ok_or_else(|| {
            anyhow::anyhow!("MTP dedicated embedding tensor `{embed_tokens_tname}` is missing")
        })?;
        ensure!(
            info.shape.len() == 2 && info.shape[1] == h,
            "{embed_tokens_tname} shape {:?} is not [vocab,{h}]",
            info.shape
        );
        let buf =
            crate::serve::forward_mlx_shared::load_gguf_qweight(gguf, mapped, &embed_tokens_tname)
                .with_context(|| {
                    format!(
                        "MTP loader expected dedicated `{embed_tokens_tname}` because \
                 mtp_use_dedicated_embeddings=True"
                    )
                })?;
        super::forward_gpu::ensure_native_embedding_admitted(&buf)
            .with_context(|| format!("admit direct execution for {embed_tokens_tname}"))?;
        super::weight_pool::register_weight_buffer(device, &buf.buffer)
            .with_context(|| format!("register {embed_tokens_tname}"))?;
        tracing::info!(
            mtp_layer = layer_index,
            mtp_use_dedicated_embeddings = true,
            tensor = %embed_tokens_tname,
            "qwen35 MTP loader: dedicated embed_tokens"
        );
        Some(buf)
    } else {
        // Belt-and-suspenders: if convert-side regression ever re-emits the dedicated
        // tensor while the flag says shared, refuse to silently ignore one of them.
        if gguf.tensor_info(&embed_tokens_tname).is_some() {
            bail!(
                "qwen35 MTP loader: mtp_use_dedicated_embeddings=False but `{embed_tokens_tname}` \
                 is present in the GGUF — convert pipeline is inconsistent. Re-emit without \
                 the dedicated tensor or set the metadata key to true."
            );
        }
        tracing::info!(
            mtp_layer = layer_index,
            mtp_use_dedicated_embeddings = false,
            "qwen35 MTP loader: sharing main token_embd (no dedicated nextn.embed_tokens)"
        );
        None
    };
    let (eh_proj, eh_proj_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{nextn}.eh_proj.weight"),
        h,
        2 * h,
        device,
    )?;
    let shared_head_norm =
        load_norm_gpu(gguf, &format!("{nextn}.shared_head_norm.weight"), h, device)?;

    // ADR-013 P14 follow-up (2026-04-30): the LM-head projection weight (`shared_head.head`)
    // follows the same shared-vs-dedicated rule as `embed_tokens`.
    //
    // Qwen3.6 27B + 35B-A3B (`mtp_use_dedicated_embeddings: False`) ship neither
    // `mtp.embed_tokens` nor `mtp.shared_head.head`; the MTP block reuses the main
    // verifier's `token_embd.weight` for the embedding lookup and resolves the
    // main output projection from `output.weight` when present, otherwise from
    // that same tied token-embedding allocation. Convert correctly skips emitting
    // `blk.{N}.nextn.shared_head_head.weight` in this configuration.
    //
    // Resolution follows metadata, not tensor-presence preference. Dedicated
    // mode requires its own head; shared mode requires that tensor to be absent
    // and borrows the resolved main head allocation. This prevents an
    // inconsistent GGUF from silently replacing the target head used by MTP.
    // `vocab_size` is derived from the exact tensor selected by that rule.
    let shared_head_head_tname = format!("{nextn}.shared_head_head.weight");
    let (shared_head_head, shared_head_head_ggml_type, vocab_size, shared_head_head_source) = if cfg
        .mtp_use_dedicated_embeddings
    {
        let info = gguf.tensor_info(&shared_head_head_tname).ok_or_else(|| {
                anyhow::anyhow!(
                    "qwen35 MTP loader: `{shared_head_head_tname}` is missing while mtp_use_dedicated_embeddings=True"
                )
            })?;
        ensure!(
            info.shape.len() == 2 && info.shape[1] == h,
            "{shared_head_head_tname} shape {:?} is not [vocab, {h}]",
            info.shape
        );
        let vocab = info.shape[0];
        let (buffer, ggml_type) =
            load_native_projection(gguf, mapped, &shared_head_head_tname, vocab, h, device)?;
        (
            buffer,
            ggml_type,
            vocab as u32,
            shared_head_head_tname.clone(),
        )
    } else {
        ensure!(
                gguf.tensor_info(&shared_head_head_tname).is_none(),
                "qwen35 MTP loader: mtp_use_dedicated_embeddings=False but `{shared_head_head_tname}` is present; refusing a silent shared-head replacement"
            );
        // Shared mode: borrow the physical main LM head. GGUF represents
        // tied Qwen3.5 heads by omitting output.weight; in that case the
        // exact token_embd.weight blocks are the authoritative head.
        let main_lm = if gguf.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        if let Some(main) = main_output_head {
            ensure!(
                    main.affine.is_none() && main.info.cols == h,
                    "qwen35 MTP loader: supplied shared output head is not a native [vocab,{h}] GGUF projection"
                );
            (
                main.buffer.clone(),
                main.info.ggml_dtype,
                main.info.rows as u32,
                main_lm.to_string(),
            )
        } else {
            let info = gguf.tensor_info(main_lm).ok_or_else(|| {
                    anyhow::anyhow!(
                        "qwen35 MTP loader: shared head missing and resolved main head {main_lm} absent"
                    )
                })?;
            ensure!(
                info.shape.len() == 2 && info.shape[1] == h,
                "{main_lm} shape {:?} is not [vocab, {h}]",
                info.shape
            );
            let vocab = info.shape[0];
            let (buffer, ggml_type) =
                load_native_projection(gguf, mapped, main_lm, vocab, h, device)?;
            (buffer, ggml_type, vocab as u32, main_lm.to_string())
        }
    };
    tracing::info!(
        mtp_layer = layer_index,
        source = %shared_head_head_source,
        vocab_size,
        "qwen35 MTP loader: shared_head_head resolved"
    );
    let attn = load_mtp_attn(gguf, mapped, cfg, layer_index, device)?;
    let (ffn, intermediate_size) = load_mtp_ffn(gguf, mapped, cfg, layer_index, device)?;

    Ok(Some(MtpWeights {
        layer_index,
        hidden_size: cfg.hidden_size,
        vocab_size,
        intermediate_size,
        loaded_tensor_names,
        enorm,
        hnorm,
        eh_proj,
        eh_proj_ggml_type,
        embed_tokens,
        shared_head_norm,
        shared_head_head,
        shared_head_head_ggml_type,
        attn,
        ffn,
    }))
}

pub(super) fn mtp_tensor_names(gguf: &GgufFile, layer_index: u32) -> Vec<String> {
    let p = format!("blk.{layer_index}.");
    let nextn = format!("{p}nextn.");
    // Inner-block tensor suffixes for both dense MTP (Qwen 3.6 27B) and MoE
    // MTP (Qwen 3.5/3.6 35B-A3B). The two FFN schemas are mutually exclusive
    // at a given block, so listing both in the membership set is safe.
    let inner = [
        // Attention (shared by both variants).
        "attn_norm.weight",
        "post_attention_norm.weight",
        "attn_q.weight",
        "attn_gate.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        // Dense FFN (Qwen 3.6 27B dense-MTP).
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        // MoE FFN (Qwen 3.5/3.6 35B-A3B MoE-MTP).
        "ffn_gate_inp.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_inp_shexp.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ];
    let mut names = Vec::new();
    for name in gguf.tensor_names() {
        if name.starts_with(&nextn) || inner.iter().any(|suffix| name == format!("{p}{suffix}")) {
            names.push(name.to_string());
        }
    }
    names.sort();
    names
}

fn load_mtp_attn(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_index: u32,
    device: &MlxDevice,
) -> Result<MtpFullAttnWeightsGpu> {
    let p = format!("blk.{layer_index}");
    let h = cfg.hidden_size as usize;
    let q_total = (cfg.num_attention_heads * cfg.head_dim) as usize;
    let kv_total = (cfg.num_key_value_heads * cfg.head_dim) as usize;
    let d = cfg.head_dim as usize;
    let attn_norm = load_norm_gpu(gguf, &format!("{p}.attn_norm.weight"), h, device)?;
    let post_attn_norm =
        load_norm_gpu(gguf, &format!("{p}.post_attention_norm.weight"), h, device)?;

    let q_name = format!("{p}.attn_q.weight");
    let q_info = gguf
        .tensor_info(&q_name)
        .ok_or_else(|| anyhow::anyhow!("{q_name} not found"))?;
    let q_gate = if q_info.shape.as_slice() == [q_total, h] {
        let (q, q_type) = load_native_projection(gguf, mapped, &q_name, q_total, h, device)?;
        let gate_name = format!("{p}.attn_gate.weight");
        if gguf.tensor_info(&gate_name).is_some() {
            let (gate, gate_type) =
                load_native_projection(gguf, mapped, &gate_name, q_total, h, device)?;
            MtpQGateWeightsGpu::Gated(FullAttnQGateWeightsGpu::Split {
                wq: q,
                wq_ggml_type: q_type,
                w_gate: gate,
                w_gate_ggml_type: gate_type,
            })
        } else {
            MtpQGateWeightsGpu::Ungated {
                wq: q,
                wq_ggml_type: q_type,
            }
        }
    } else if q_info.shape.as_slice() == [2 * q_total, h] {
        let (fused, q_type) =
            load_native_projection(gguf, mapped, &q_name, 2 * q_total, h, device)?;
        MtpQGateWeightsGpu::Gated(FullAttnQGateWeightsGpu::Fused {
            weight: fused,
            ggml_type: q_type,
        })
    } else {
        bail!(
            "{q_name} shape {:?}, expected [{q_total},{h}] or [{}, {h}] interleaved Q+gate",
            q_info.shape,
            2 * q_total,
        );
    };

    let (wk, wk_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_k.weight"),
        kv_total,
        h,
        device,
    )?;
    let (wv, wv_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_v.weight"),
        kv_total,
        h,
        device,
    )?;
    let (wo, wo_ggml_type) = load_native_projection(
        gguf,
        mapped,
        &format!("{p}.attn_output.weight"),
        h,
        q_total,
        device,
    )?;

    Ok(MtpFullAttnWeightsGpu {
        attn_norm,
        post_attn_norm,
        q_gate,
        wk,
        wk_ggml_type,
        wv,
        wv_ggml_type,
        attn_q_norm: load_norm_gpu(gguf, &format!("{p}.attn_q_norm.weight"), d, device)?,
        attn_k_norm: load_norm_gpu(gguf, &format!("{p}.attn_k_norm.weight"), d, device)?,
        wo,
        wo_ggml_type,
    })
}

fn load_mtp_ffn(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_index: u32,
    device: &MlxDevice,
) -> Result<(MtpFfnWeightsGpu, u32)> {
    let p = format!("blk.{layer_index}");
    let has_dense = gguf.tensor_info(&format!("{p}.ffn_gate.weight")).is_some();
    let has_moe = gguf
        .tensor_info(&format!("{p}.ffn_gate_exps.weight"))
        .is_some();
    match (has_dense, has_moe) {
        (true, false) => load_mtp_dense_ffn(gguf, mapped, cfg, &p, device),
        (false, true) => load_mtp_moe_ffn(gguf, mapped, cfg, layer_index, &p, device),
        (true, true) => bail!(
            "qwen35 MTP loader: block {layer_index} has BOTH dense (`{p}.ffn_gate.weight`) and \
             MoE (`{p}.ffn_gate_exps.weight`) FFN tensors — GGUF is malformed"
        ),
        (false, false) => bail!(
            "qwen35 MTP loader: block {layer_index} has NEITHER dense (`{p}.ffn_gate.weight`) \
             nor MoE (`{p}.ffn_gate_exps.weight`) inner FFN tensors"
        ),
    }
}

fn load_mtp_dense_ffn(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    p: &str,
    device: &MlxDevice,
) -> Result<(MtpFfnWeightsGpu, u32)> {
    let layer_idx = cfg.num_hidden_layers;
    let (gate_type, up_type, down_type) = dense_ffn_tensor_types(gguf, layer_idx)?;
    match dense_ffn_storage(layer_idx, gate_type, up_type, down_type)? {
        DenseFfnStorage::Quantized => {
            let weights_q = load_dense_ffn_quantized(gguf, mapped, layer_idx, cfg, device)
                .with_context(|| format!("MTP native dense FFN {p}"))?;
            let intermediate_size = weights_q.intermediate_size;
            let dense_gpu = DenseFfnWeightsGpuQ::from_quantized(&weights_q);
            Ok((
                MtpFfnWeightsGpu::DenseQ { weights: dense_gpu },
                intermediate_size,
            ))
        }
        DenseFfnStorage::NativeScalar => {
            let weights_native = load_dense_ffn_native(gguf, mapped, layer_idx, cfg, device)
                .with_context(|| format!("MTP native scalar dense FFN {p}"))?;
            let intermediate_size = weights_native.intermediate_size;
            let weights = DenseFfnWeightsGpu::from_native(&weights_native);
            Ok((
                MtpFfnWeightsGpu::Dense {
                    weights,
                    intermediate_size,
                },
                intermediate_size,
            ))
        }
    }
}

fn load_mtp_moe_ffn(
    gguf: &GgufFile,
    mapped: &GgufMappedTensorSet<'_>,
    cfg: &Qwen35Config,
    layer_index: u32,
    p: &str,
    device: &MlxDevice,
) -> Result<(MtpFfnWeightsGpu, u32)> {
    let moe_cfg = cfg.moe.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "qwen35 MTP loader: block {layer_index} has MoE FFN tensors (`{p}.ffn_gate_exps.weight`) \
             but `cfg.moe` is None — the model metadata is inconsistent"
        )
    })?;
    // Load with the same quantized path used by every other MoE layer; this
    // keeps expert weights as native GGML blocks (no F32 expansion) and
    // matches what the verifier's main forward path consumes.
    let weights_q = load_moe_ffn_quantized(gguf, mapped, cfg, layer_index, device)
        .with_context(|| format!("MTP MoE FFN layer {layer_index}"))?;
    let moe_gpu = MoeFfnWeightsGpuQ::from_quantized(
        &weights_q,
        moe_cfg.num_experts,
        moe_cfg.moe_intermediate_size,
        cfg.hidden_size,
    )
    .with_context(|| format!("MTP MoE retain layer {layer_index}"))?;
    let shape = MoeFfnShape {
        hidden_size: cfg.hidden_size,
        num_experts: moe_cfg.num_experts,
        num_experts_per_tok: moe_cfg.num_experts_per_tok,
        moe_intermediate_size: moe_cfg.moe_intermediate_size,
        shared_intermediate_size: moe_cfg.shared_expert_intermediate_size,
    };
    Ok((
        MtpFfnWeightsGpu::Moe {
            weights: moe_gpu,
            shape,
        },
        moe_cfg.moe_intermediate_size,
    ))
}

fn load_norm_gpu(gguf: &GgufFile, name: &str, len: usize, device: &MlxDevice) -> Result<MlxBuffer> {
    let data = load_f32_tensor(gguf, name, device).with_context(|| name.to_string())?;
    ensure!(data.len() == len, "{name} length {} != {len}", data.len());
    // W-5b.7 iter 2: MTP norm weights are static / reused across decode tokens —
    // route them through the residency-aware helper.
    upload_f32_weight(&data, device).with_context(|| format!("upload {name}"))
}
