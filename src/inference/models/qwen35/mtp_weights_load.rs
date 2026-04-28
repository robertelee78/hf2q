use anyhow::{bail, ensure, Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::{MlxBuffer, MlxDevice};

use super::ffn::DenseFfnWeights;
use super::gpu_ffn::{DenseFfnWeightsGpu};
use super::gpu_full_attn::{upload_bf16_from_f32, upload_f32_weight};
use super::mtp::{MtpFullAttnWeightsGpu, MtpWeights};
use super::weight_loader::load_f32_tensor;
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
    let embed_tokens = upload_bf16_required(gguf, &format!("{nextn}.embed_tokens.weight"), device)?;
    let (eh_proj_embed, eh_proj_hidden) =
        load_split_eh_proj(gguf, &format!("{nextn}.eh_proj.weight"), h, device)?;
    let shared_head_norm =
        load_norm_gpu(gguf, &format!("{nextn}.shared_head_norm.weight"), h, device)?;
    let shared_head_head_f32 =
        load_f32_tensor(gguf, &format!("{nextn}.shared_head_head.weight"), device)
            .with_context(|| format!("{nextn}.shared_head_head.weight"))?;
    ensure!(
        shared_head_head_f32.len() % h == 0,
        "{nextn}.shared_head_head.weight has {} floats, not divisible by hidden_size {h}",
        shared_head_head_f32.len()
    );
    let vocab_size = (shared_head_head_f32.len() / h) as u32;
    let shared_head_head = upload_bf16_from_f32(&shared_head_head_f32, device)
        .context("MTP upload shared_head_head")?;
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
    let inner = [
        "attn_norm.weight",
        "post_attention_norm.weight",
        "attn_q.weight",
        "attn_gate.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
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
) -> Result<(DenseFfnWeightsGpu, u32)> {
    let p = format!("blk.{layer_index}");
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
    Ok((
        DenseFfnWeightsGpu::from_cpu(&weights, device).context("MTP upload dense FFN")?,
        intermediate as u32,
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
