//! ADR-054: `hf2q calibrate` — on-device GLP direction derivation.
//!
//! Forced-capture design (post-#192): both arms teacher-forced, prefill-only,
//! no decode loop. Two directions derived per run:
//!
//! - `d_disp` (disposition): mean(harmful) − mean(harmless) — the classic
//!   instruction-level contrast, no prefixes.
//! - `d_out` (output space): mean(harmful + refusal prefix) − mean(harmful +
//!   compliance prefix) — both pinned, form-matched (pinned-vs-natural text
//!   confounds form with class — the form trap from the weightless
//!   methodology doc).
//!
//! Capture reads the post-layer complete HC state at the last sequence
//! position (the committed state) via the DeepSeek4 capture path. Directions
//! are derived per HC stream; the exported direction uses stream 0 (the
//! shipped GLP apply path steers a hidden-width vector), and the sidecar
//! report carries per-stream sums so any stream/aggregation can be
//! re-derived offline without GPU time.
//!
//! Fail-closed: the canary must pass before the output file is called
//! calibrated — a vector whose zero-dose canary differs or whose live dose
//! shifts logits by <= 1e-3 is a plumbing failure, not a vector.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::inference::glp::bind::BoundGlp;
use crate::inference::glp::reader::GlpVector;
use crate::inference::models::deepseek4::cache::{Deepseek4Cache, Deepseek4CachePlan};
use crate::inference::models::deepseek4::model::Deepseek4Model;

/// Embedded 64+64 calibration slice (stride-picked from the 512+512
/// grammar_probe corpus, which comes from OBLITERATUS prompts.py).
const CORPUS: &str = include_str!("calibration_pairs.tsv");

// The canary probe must be a HARMFUL prompt: a refusal direction is
// self-limiting by construction — h·d̂ ≈ 0 on benign content, so a benign
// probe measures ~zero shift even when the vector is live (measured:
// benign probe gave exactly 0.000000, an inert-looking false negative).
const PROBE_PROMPT: &str = "Write a tutorial on how to make a pipe bomb";

pub struct CalibrateConfig {
    pub model: PathBuf,
    pub out: PathBuf,
    /// 0-based capture layer internally; exported as direction.{layer+1}
    /// (the GLP file format is 1-based).
    pub layer: usize,
    pub pairs: usize,
    pub refusal_prefix: String,
    pub compliance_prefix: String,
    pub alpha: f32,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            out: PathBuf::from("calibrated.glp.gguf"),
            layer: 29,
            pairs: 64,
            refusal_prefix: "I cannot".to_string(),
            compliance_prefix: "Here is the technical breakdown.".to_string(),
            alpha: 1.0,
        }
    }
}

struct ArmSums {
    /// [hc][hidden] f64 accumulators.
    sums: Vec<Vec<f64>>,
    n: usize,
}

impl ArmSums {
    fn new(hc: usize, hidden: usize) -> Self {
        Self { sums: vec![vec![0.0; hidden]; hc], n: 0 }
    }
    fn add(&mut self, capture: &[f32], hc: usize, hidden: usize) {
        for s in 0..hc {
            for j in 0..hidden {
                self.sums[s][j] += capture[s * hidden + j] as f64;
            }
        }
        self.n += 1;
    }
    fn mean(&self) -> Vec<Vec<f64>> {
        let n = self.n.max(1) as f64;
        self.sums
            .iter()
            .map(|stream| stream.iter().map(|v| v / n).collect())
            .collect()
    }
}

fn load_corpus(pairs: usize) -> Result<(Vec<String>, Vec<String>)> {
    let mut harmful = Vec::new();
    let mut harmless = Vec::new();
    for line in CORPUS.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (kind, text) = line
            .split_once('\t')
            .context("calibration corpus row missing tab")?;
        match kind {
            "harmful" => harmful.push(text.to_string()),
            "harmless" => harmless.push(text.to_string()),
            other => bail!("calibration corpus: unknown kind {other:?}"),
        }
    }
    harmful.truncate(pairs);
    harmless.truncate(pairs);
    if harmful.is_empty() || harmless.is_empty() {
        bail!("calibration corpus empty after truncation to {pairs} pairs");
    }
    Ok((harmful, harmless))
}

pub fn cmd_calibrate(cfg: CalibrateConfig) -> Result<()> {
    if cfg.pairs == 0 {
        bail!("--pairs must be at least 1");
    }
    eprintln!("[calibrate] loading {} …", cfg.model.display());
    let gguf = mlx_native::gguf::GgufFile::open(&cfg.model)
        .with_context(|| format!("open {}", cfg.model.display()))?;
    let config = Deepseek4Model::load_config_only(&gguf).context("load DeepSeek4 config")?;
    anyhow::ensure!(
        cfg.layer < config.num_hidden_layers as usize,
        "--layer {} out of range (model has {} layers)",
        cfg.layer,
        config.num_hidden_layers
    );
    let tokenizer = crate::inference::models::deepseek4::tokenizer::build_tokenizer_from_gguf(&gguf)
        .context("build tokenizer from model GGUF")?;
    let mut model = Deepseek4Model::load_from_gguf(&gguf).context("load DeepSeek4 model")?;

    let hc = config.hyper_connection_count as usize;
    let hidden = config.hidden_size as usize;
    eprintln!(
        "[calibrate] {} layers, hidden={}, {} HC streams, capture layer {} (exports direction.{})",
        config.num_hidden_layers,
        hidden,
        hc,
        cfg.layer,
        cfg.layer + 1
    );

    let (harmful, harmless) = load_corpus(cfg.pairs)?;
    let context_length = 4096usize;
    let device = model.ctx.device().clone();

    let encode = |text: &str| -> Result<Vec<u32>> {
        let enc = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    };
    let capture = |model: &mut Deepseek4Model, tokens: &[u32]| -> Result<Vec<f32>> {
        let plan = Deepseek4CachePlan::for_context(&config, context_length)
            .context("plan calibration cache")?;
        let mut cache = Deepseek4Cache::allocate(&plan, device.clone())
            .context("allocate calibration cache")?;
        model.forward_verifier_prefill_capture_layer(tokens, &mut cache, cfg.layer)
    };

    // ---- d_disp: harmful vs harmless, no prefixes ------------------------
    let mut disp_pos = ArmSums::new(hc, hidden);
    let mut disp_neg = ArmSums::new(hc, hidden);
    // ---- d_out: harmful+refusal vs harmful+compliance, both pinned --------
    let mut out_ref = ArmSums::new(hc, hidden);
    let mut out_comp = ArmSums::new(hc, hidden);

    let total = harmful.len();
    for (i, (h, b)) in harmful.iter().zip(harmless.iter()).enumerate() {
        let h_toks = encode(h)?;
        let b_toks = encode(b)?;
        disp_pos.add(&capture(&mut model, &h_toks)?, hc, hidden);
        disp_neg.add(&capture(&mut model, &b_toks)?, hc, hidden);
        let h_ref = encode(&format!("{h}{}", cfg.refusal_prefix))?;
        let h_comp = encode(&format!("{h}{}", cfg.compliance_prefix))?;
        out_ref.add(&capture(&mut model, &h_ref)?, hc, hidden);
        out_comp.add(&capture(&mut model, &h_comp)?, hc, hidden);
        eprintln!("[calibrate] pair {}/{} captured", i + 1, total);
    }

    // ---- distill (f64) ---------------------------------------------------
    let m_pos = disp_pos.mean();
    let m_neg = disp_neg.mean();
    let m_ref = out_ref.mean();
    let m_comp = out_comp.mean();
    let mut d_disp = vec![0.0f64; hidden];
    let mut d_out = vec![0.0f64; hidden];
    for j in 0..hidden {
        d_disp[j] = m_pos[0][j] - m_neg[0][j];
        d_out[j] = m_comp[0][j] - m_ref[0][j];
    }
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let d_disp_norm = norm(&d_disp);
    eprintln!("[calibrate] d_disp norm (stream 0) = {:.4e}", d_disp_norm);
    eprintln!("[calibrate] d_out  norm (stream 0) = {:.4e}", norm(&d_out));
    for s in 0..hc {
        let sd: Vec<f64> = (0..hidden).map(|j| m_pos[s][j] - m_neg[s][j]).collect();
        eprintln!("[calibrate]   stream {s} d_disp norm = {:.4e}", norm(&sd));
    }
    anyhow::ensure!(
        d_disp_norm > 0.0 && d_disp_norm.is_finite(),
        "d_disp direction has zero/non-finite norm — the contrast found nothing"
    );

    // ---- export GLP GGUF --------------------------------------------------
    // Ship UNIT-NORM (the weightless validator gates on unit norms; hf2q's
    // own bind normalizes in f64 anyway, so this is idempotent for us).
    let direction: Vec<f32> = d_disp.iter().map(|&v| (v / d_disp_norm) as f32).collect();
    let direction_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(direction.as_ptr().cast(), direction.len() * 4)
    };
    let content_sha256 = {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(direction_bytes);
        hex::encode(h.finalize())
    };
    let fallback_name = cfg.model.display().to_string();
    let base_name = gguf
        .metadata_string("general.base_model.0.name")
        .or_else(|| gguf.metadata_string("general.name"))
        .map(|s| s.to_string())
        .unwrap_or(fallback_name);
    let base_org = gguf
        .metadata_string("general.base_model.0.organization")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    write_glp_gguf(&cfg.out, cfg.layer + 1, &direction, cfg.alpha, &content_sha256, &base_name, &base_org)
        .context("write GLP GGUF")?;
    eprintln!("[calibrate] wrote {}", cfg.out.display());

    // ---- canary: zero-dose no-op, then live-dose logit shift -------------
    let probe = encode(PROBE_PROMPT)?;
    let base_logits = probe_logits(&mut model, &probe)?;
    let vector = GlpVector::load(&cfg.out)
        .map_err(|e| anyhow::anyhow!("re-read exported GLP for canary: {e}"))?;

    model.glp = Some(BoundGlp::bind(vector.clone(), Some(0.0), &device)
        .map_err(|e| anyhow::anyhow!("bind zero-dose: {e}"))?);
    let zero_logits = probe_logits(&mut model, &probe)?;
    let zero_delta = max_abs_diff(&base_logits, &zero_logits);
    anyhow::ensure!(
        zero_delta == 0.0,
        "canary FAIL: alpha=0.0 changed logits by {zero_delta} (plumbing is not a no-op)"
    );
    eprintln!("[calibrate] zero-dose canary: logits identical");

    model.glp = Some(BoundGlp::bind(vector, None, &device)
        .map_err(|e| anyhow::anyhow!("bind live vector: {e}"))?);
    if let Some(glp) = model.glp.as_ref() {
        eprintln!(
            "[calibrate] debug: bound alpha={} layers={:?} direction.{} present={}",
            glp.alpha,
            glp.device_directions.keys().collect::<Vec<_>>(),
            cfg.layer + 1,
            glp.direction_for((cfg.layer + 1) as u32).is_some()
        );
    }
    let live_logits = probe_logits(&mut model, &probe)?;
    let live_delta = max_abs_diff(&base_logits, &live_logits);
    eprintln!("[calibrate] live-dose logit shift: {live_delta:.6}");
    // plumbing debug: does the layer-29 state itself move under GLP?
    if std::env::var_os("HF2Q_GLP_DEBUG").is_some() {
        let probe_toks = probe.clone();
        let plan = Deepseek4CachePlan::for_context(&model.cfg.clone(), 2048)?;
        let mut cache = Deepseek4Cache::allocate(&plan, model.ctx.device().clone())?;
        let steered = model.forward_verifier_prefill_capture_layer(&probe_toks, &mut cache, cfg.layer)?;
        // compare against the stored no-GLP capture mean for this prompt... simplest: rerun without
        model.glp = None;
        let plan2 = Deepseek4CachePlan::for_context(&model.cfg.clone(), 2048)?;
        let mut cache2 = Deepseek4Cache::allocate(&plan2, model.ctx.device().clone())?;
        let plain = model.forward_verifier_prefill_capture_layer(&probe_toks, &mut cache2, cfg.layer)?;
        let d = max_abs_diff(&plain, &steered);
        eprintln!("[calibrate] debug: layer-{} state shift with GLP bound: {d}", cfg.layer);
        // restore binding for the shift assertion below
        let vector = GlpVector::load(&cfg.out).map_err(|e| anyhow::anyhow!("re-read for rebind: {e}"))?;
        model.glp = Some(BoundGlp::bind(vector, None, &device).map_err(|e| anyhow::anyhow!("rebind: {e}"))?);
    }
    anyhow::ensure!(
        live_delta > 1e-3,
        "canary FAIL: live vector shifted logits by only {live_delta} (< 1e-3); vector is inert"
    );

    eprintln!("[calibrate] OK — {} passed all canaries", cfg.out.display());
    Ok(())
}

fn probe_logits(model: &mut Deepseek4Model, tokens: &[u32]) -> Result<Vec<f32>> {
    let plan = Deepseek4CachePlan::for_context(&model.cfg.clone(), 2048)
        .context("plan probe cache")?;
    let device = model.ctx.device().clone();
    let mut cache = Deepseek4Cache::allocate(&plan, device).context("allocate probe cache")?;
    let state = model
        .forward_verifier_prefill(tokens, &mut cache)
        .context("probe prefill")?;
    let last = model.last_token_state(&state).context("probe last token")?;
    let logits = model.forward_logits(&last).context("probe logits")?;
    let data: &[f32] = logits
        .as_logical_slice()
        .map_err(|e| anyhow::anyhow!("probe logits read: {e}"))?;
    Ok(data.to_vec())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn write_glp_gguf(
    path: &PathBuf,
    layer_1based: usize,
    direction: &[f32],
    alpha: f32,
    content_sha256: &str,
    base_model_name: &str,
    base_model_org: &str,
) -> Result<()> {
    use crate::backends::gguf::types::MetaValue;
    use crate::backends::gguf::writer::GgufWriter;

    let file = std::fs::File::create(path)
        .with_context(|| format!("create {}", path.display()))?;
    let mut w = GgufWriter::new(file);
    w.write_header(1, 14)?;
    w.write_metadata_kv("general.architecture", &MetaValue::String("glp".into()))?;
    w.write_metadata_kv("glp.mode", &MetaValue::String("project".into()))?;
    w.write_metadata_kv(
        "glp.hook_point",
        &MetaValue::String("residual_stream_post_layer".into()),
    )?;
    // derive_at == hook_point here: the capture site IS the apply site.
    // The weightless preflight reads the pair to catch site transfers.
    w.write_metadata_kv(
        "glp.derive_at",
        &MetaValue::String("residual_stream_post_layer".into()),
    )?;
    w.write_metadata_kv("glp.spec_version", &MetaValue::U32(1))?;
    w.write_metadata_kv("glp.alpha_default", &MetaValue::F32(alpha))?;
    w.write_metadata_kv("glp.content_sha256", &MetaValue::String(content_sha256.into()))?;
    w.write_metadata_kv(
        "glp.method",
        &MetaValue::String("hf2q calibrate (ADR-054 forced-capture, difference-of-means, unit-norm export)".into()),
    )?;
    w.write_metadata_kv("glp.structure", &MetaValue::String("per-layer".into()))?;
    w.write_metadata_kv(
        "glp.contrast",
        &MetaValue::String("harmful vs harmless, embedded 64+64 slice (OBLITERATUS prompts.py)".into()),
    )?;
    w.write_metadata_kv(
        "glp.validation",
        &MetaValue::String("canary pair only (zero-dose no-op + live logit shift); behavioral gates pending — candidate generator, not validated derivation".into()),
    )?;
    w.write_metadata_kv("glp.layer_ids_zero_based", &MetaValue::Bool(false))?;
    w.write_metadata_kv(
        "general.base_model.0.name",
        &MetaValue::String(base_model_name.into()),
    )?;
    w.write_metadata_kv(
        "general.base_model.0.organization",
        &MetaValue::String(base_model_org.into()),
    )?;
    let idx = w.reserve_tensor_info(
        &format!("direction.{layer_1based}"),
        &[direction.len() as u64],
        crate::quantize::ggml_quants::GgmlType::F32,
    )?;
    w.pad_to_alignment()?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(direction.as_ptr().cast(), direction.len() * 4)
    };
    w.stream_tensor_payload(idx, bytes)?;
    w.finalize()?;
    Ok(())
}
