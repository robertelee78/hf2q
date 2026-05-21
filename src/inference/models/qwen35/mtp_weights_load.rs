use anyhow::{bail, ensure, Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::{MlxBuffer, MlxDevice};

use super::ffn::{DenseFfnWeights, MoeFfnShape};
use super::gpu_ffn::{DenseFfnWeightsGpu, MoeFfnWeightsGpuQ};
use super::gpu_full_attn::{upload_bf16_from_f32, upload_f32_weight, upload_q4_0_from_f32};
use super::mtp::{MtpFfnWeightsGpu, MtpFullAttnWeightsGpu, MtpWeights};
use super::weight_loader::{load_f32_tensor, load_moe_ffn_quantized};
use super::Qwen35Config;

pub fn load_mtp_weights_if_present(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    device: &MlxDevice,
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
        let buf = upload_bf16_required(gguf, &embed_tokens_tname, device)
            .with_context(|| {
                format!(
                    "MTP loader expected dedicated `{embed_tokens_tname}` because \
                     mtp_use_dedicated_embeddings=True (set explicitly via metadata or \
                     inferred from tensor presence); to share main embeddings, re-emit \
                     the GGUF without `nextn.embed_tokens.weight` or set the metadata \
                     key `{p}.nextn.use_dedicated_embeddings = false`"
                )
            })?;
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
    let (eh_proj_embed, eh_proj_hidden) =
        load_split_eh_proj(gguf, &format!("{nextn}.eh_proj.weight"), h, device)?;
    let shared_head_norm =
        load_norm_gpu(gguf, &format!("{nextn}.shared_head_norm.weight"), h, device)?;

    // ADR-013 P14 follow-up (2026-04-30): the LM-head projection weight (`shared_head.head`)
    // follows the same shared-vs-dedicated rule as `embed_tokens`.
    //
    // Qwen3.6 27B + 35B-A3B (`mtp_use_dedicated_embeddings: False`) ship neither
    // `mtp.embed_tokens` nor `mtp.shared_head.head`; the MTP block reuses the main
    // verifier's `token_embd.weight` for the embedding lookup AND `output.weight`
    // for the final logit projection. Convert correctly skips emitting
    // `blk.{N}.nextn.shared_head_head.weight` in this configuration.
    //
    // Resolution: if the dedicated tensor is present we use it (Qwen3.5 MTP);
    // otherwise (shared mode) we fall back to the main `output.weight`. The
    // bf16 GPU buffer is materialised the same way either path; vocab_size
    // is derived from the row count of whichever tensor we actually loaded.
    let shared_head_head_tname = format!("{nextn}.shared_head_head.weight");
    let (shared_head_head_f32, shared_head_head_source) =
        if gguf.tensor_info(&shared_head_head_tname).is_some() {
            let data = load_f32_tensor(gguf, &shared_head_head_tname, device)
                .with_context(|| shared_head_head_tname.clone())?;
            (data, shared_head_head_tname.clone())
        } else if !cfg.mtp_use_dedicated_embeddings {
            // Shared mode: borrow the main LM head.
            let main_lm = "output.weight";
            ensure!(
                gguf.tensor_info(main_lm).is_some(),
                "qwen35 MTP loader: mtp_use_dedicated_embeddings=False and \
                 `{shared_head_head_tname}` absent, but main `{main_lm}` is also \
                 missing — cannot resolve the MTP final projection."
            );
            let data = load_f32_tensor(gguf, main_lm, device)
                .with_context(|| format!("MTP shared_head_head fallback to {main_lm}"))?;
            (data, main_lm.to_string())
        } else {
            bail!(
                "qwen35 MTP loader: `{shared_head_head_tname}` is missing AND \
                 mtp_use_dedicated_embeddings=True — cannot resolve the MTP final \
                 projection. Re-emit the GGUF with the dedicated tensor or set the \
                 flag to false."
            );
        };
    ensure!(
        shared_head_head_f32.len() % h == 0,
        "{shared_head_head_source} has {} floats, not divisible by hidden_size {h}",
        shared_head_head_f32.len()
    );
    let vocab_size = (shared_head_head_f32.len() / h) as u32;
    tracing::info!(
        mtp_layer = layer_index,
        source = %shared_head_head_source,
        vocab_size,
        "qwen35 MTP loader: shared_head_head resolved"
    );
    // ADR-028 iter-157: shared_head_head was stored as F32→BF16 (2.54 GB at
    // qwen3.6 vocab=248320 × hidden=5120). At 587 GB/s peak bandwidth that
    // pays 4.33 ms per draft step — matched iter-156 measurement (4.52 ms,
    // 65% of forward_draft). Verifier already learned this lesson at
    // forward_gpu.rs:416-422 ("Q4_0 matmul on Apple Silicon is faster than
    // BF16 at BOTH M=1 and M>1, ~1.4 ms saved per step → +14 tok/s") and
    // switched its lm_head from BF16 to Q4_0. Replicate that here for the
    // MTP shared_head_head — same `apply_linear_projection_f32` call site
    // works with the Q4_0 U8 buffer (verified by reading verifier path at
    // forward_gpu.rs:856).
    let shared_head_head = upload_q4_0_from_f32(&shared_head_head_f32, device)
        .context("MTP upload shared_head_head Q4_0")?;
    let attn = load_mtp_attn(gguf, cfg, layer_index, device)?;
    let (ffn, intermediate_size) = load_mtp_ffn(gguf, cfg, layer_index, device)?;

    Ok(Some(MtpWeights {
        layer_index,
        hidden_size: cfg.hidden_size,
        vocab_size,
        intermediate_size,
        loaded_tensor_names,
        enorm,
        hnorm,
        eh_proj_embed,
        eh_proj_hidden,
        embed_tokens,
        shared_head_norm,
        shared_head_head,
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

    let q_or_q_gate = load_f32_tensor(gguf, &format!("{p}.attn_q.weight"), device)
        .with_context(|| format!("{p}.attn_q.weight"))?;
    let (wq_f32, w_gate_f32) = if q_or_q_gate.len() == q_total * h {
        let gate_name = format!("{p}.attn_gate.weight");
        let gate = if gguf.tensor_info(&gate_name).is_some() {
            let gate = load_f32_tensor(gguf, &gate_name, device).with_context(|| gate_name.clone())?;
            ensure!(gate.len() == q_total * h, "{gate_name} shape mismatch");
            Some(gate)
        } else {
            None
        };
        (q_or_q_gate, gate)
    } else if q_or_q_gate.len() == 2 * q_total * h {
        split_interleaved_q_gate(&q_or_q_gate, cfg)?
    } else {
        bail!(
            "{p}.attn_q.weight has {} floats, expected {} (Q) or {} (interleaved Q+gate)",
            q_or_q_gate.len(),
            q_total * h,
            2 * q_total * h
        );
    };

    let wk_f32 = load_f32_tensor(gguf, &format!("{p}.attn_k.weight"), device)
        .with_context(|| format!("{p}.attn_k.weight"))?;
    let wv_f32 = load_f32_tensor(gguf, &format!("{p}.attn_v.weight"), device)
        .with_context(|| format!("{p}.attn_v.weight"))?;
    let wo_f32 = load_f32_tensor(gguf, &format!("{p}.attn_output.weight"), device)
        .with_context(|| format!("{p}.attn_output.weight"))?;
    ensure!(wk_f32.len() == kv_total * h, "{p}.attn_k.weight shape mismatch");
    ensure!(wv_f32.len() == kv_total * h, "{p}.attn_v.weight shape mismatch");
    ensure!(wo_f32.len() == h * q_total, "{p}.attn_output.weight shape mismatch");

    Ok(MtpFullAttnWeightsGpu {
        attn_norm,
        post_attn_norm,
        wq: upload_bf16_from_f32(&wq_f32, device).context("MTP upload wq")?,
        wk: upload_bf16_from_f32(&wk_f32, device).context("MTP upload wk")?,
        wv: upload_bf16_from_f32(&wv_f32, device).context("MTP upload wv")?,
        w_gate: match w_gate_f32 {
            Some(w) => Some(upload_bf16_from_f32(&w, device).context("MTP upload w_gate")?),
            None => None,
        },
        attn_q_norm: load_norm_gpu(gguf, &format!("{p}.attn_q_norm.weight"), d, device)?,
        attn_k_norm: load_norm_gpu(gguf, &format!("{p}.attn_k_norm.weight"), d, device)?,
        wo: upload_bf16_from_f32(&wo_f32, device).context("MTP upload wo")?,
    })
}

fn load_mtp_ffn(
    gguf: &GgufFile,
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
        (true, false) => load_mtp_dense_ffn(gguf, cfg, &p, device),
        (false, true) => load_mtp_moe_ffn(gguf, cfg, layer_index, &p, device),
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
    cfg: &Qwen35Config,
    p: &str,
    device: &MlxDevice,
) -> Result<(MtpFfnWeightsGpu, u32)> {
    let h = cfg.hidden_size as usize;
    let gate = load_f32_tensor(gguf, &format!("{p}.ffn_gate.weight"), device)
        .with_context(|| format!("{p}.ffn_gate.weight"))?;
    ensure!(gate.len() % h == 0, "{p}.ffn_gate.weight width mismatch");
    let intermediate = gate.len() / h;
    let up = load_f32_tensor(gguf, &format!("{p}.ffn_up.weight"), device)
        .with_context(|| format!("{p}.ffn_up.weight"))?;
    let down = load_f32_tensor(gguf, &format!("{p}.ffn_down.weight"), device)
        .with_context(|| format!("{p}.ffn_down.weight"))?;
    ensure!(up.len() == intermediate * h, "{p}.ffn_up.weight shape mismatch");
    ensure!(down.len() == h * intermediate, "{p}.ffn_down.weight shape mismatch");
    let weights = DenseFfnWeights { gate, up, down };
    let dense_gpu =
        DenseFfnWeightsGpu::from_cpu(&weights, device).context("MTP upload dense FFN")?;
    let intermediate_size = intermediate as u32;
    Ok((
        MtpFfnWeightsGpu::Dense {
            weights: dense_gpu,
            intermediate_size,
        },
        intermediate_size,
    ))
}

fn load_mtp_moe_ffn(
    gguf: &GgufFile,
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
    let weights_q = load_moe_ffn_quantized(gguf, layer_index, device)
        .with_context(|| format!("MTP MoE FFN layer {layer_index}"))?;
    let moe_gpu = MoeFfnWeightsGpuQ::from_quantized(
        weights_q.expert_gate_q.clone(),
        weights_q.expert_up_q.clone(),
        weights_q.expert_down_q.clone(),
        weights_q.ggml_type_gate_up,
        weights_q.ggml_type_down,
        moe_cfg.num_experts,
        moe_cfg.moe_intermediate_size,
        cfg.hidden_size,
        &weights_q.router,
        &weights_q.shared_gate_logit,
        &weights_q.shared_gate,
        &weights_q.shared_up,
        &weights_q.shared_down,
        device,
    )
    .with_context(|| format!("MTP MoE upload layer {layer_index}"))?;
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

fn upload_bf16_required(gguf: &GgufFile, name: &str, device: &MlxDevice) -> Result<MlxBuffer> {
    let data = load_f32_tensor(gguf, name, device).with_context(|| name.to_string())?;
    upload_bf16_from_f32(&data, device).with_context(|| format!("upload {name}"))
}

fn load_split_eh_proj(
    gguf: &GgufFile,
    name: &str,
    hidden_size: usize,
    device: &MlxDevice,
) -> Result<(MlxBuffer, MlxBuffer)> {
    let data = load_f32_tensor(gguf, name, device).with_context(|| name.to_string())?;
    ensure!(data.len() == hidden_size * hidden_size * 2, "{name} shape mismatch");
    let mut embed = vec![0.0f32; hidden_size * hidden_size];
    let mut hidden = vec![0.0f32; hidden_size * hidden_size];
    for row in 0..hidden_size {
        let src = row * 2 * hidden_size;
        let dst = row * hidden_size;
        embed[dst..dst + hidden_size].copy_from_slice(&data[src..src + hidden_size]);
        hidden[dst..dst + hidden_size]
            .copy_from_slice(&data[src + hidden_size..src + 2 * hidden_size]);
    }
    Ok((
        upload_bf16_from_f32(&embed, device).context("upload MTP eh_proj embed half")?,
        upload_bf16_from_f32(&hidden, device).context("upload MTP eh_proj hidden half")?,
    ))
}

fn split_interleaved_q_gate(data: &[f32], cfg: &Qwen35Config) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let h = cfg.hidden_size as usize;
    let nh = cfg.num_attention_heads as usize;
    let d = cfg.head_dim as usize;
    let q_total = nh * d;
    ensure!(data.len() == 2 * q_total * h, "interleaved Q+gate shape mismatch");
    let mut q = vec![0.0f32; q_total * h];
    let mut gate = vec![0.0f32; q_total * h];
    for head_idx in 0..nh {
        let src_q = (head_idx * 2 * d) * h;
        let src_gate = ((head_idx * 2 + 1) * d) * h;
        let dst = head_idx * d * h;
        q[dst..dst + d * h].copy_from_slice(&data[src_q..src_q + d * h]);
        gate[dst..dst + d * h].copy_from_slice(&data[src_gate..src_gate + d * h]);
    }
    Ok((q, Some(gate)))
}
