//! ADR-054: `hf2q calibrate` — on-device GLP direction derivation,
//! generalized to every served GLP family (#300).
//!
//! Forced-capture design (post-#192): both arms teacher-forced,
//! prefill-only, no decode loop. Two directions derived per run:
//!
//! - `d_disp` (disposition): mean(harmful) − mean(harmless) — the classic
//!   instruction-level contrast, no prefixes. THE export arm (gate-2
//!   resolution).
//! - `d_out` (output space): mean(harmful + refusal prefix) − mean(harmful +
//!   compliance prefix) — both pinned, form-matched (pinned-vs-natural
//!   text confounds form with class). Report-only register instrument.
//!
//! The pipeline skeleton here is family-agnostic: corpus, contrastive
//! arms, distill, export, canary, register report, sidecar. The family
//! adapter (`family.rs` + `deepseek4.rs` + `qwen35.rs`) supplies the
//! capture read arms, tokenizer, forward plumbing, and the canary bind
//! contract. The capture/apply site pair is a first-class dial
//! (`--hook-point` / `--capture-site`); the defaults are the family's
//! published pair, and no site pair becomes an export default except a
//! measured ladder winner (ADR-054 Deferred follow-ups item 4).
//!
//! Fail-closed: the canary must pass before the output file is called
//! calibrated — a vector whose zero-dose canary differs or whose live
//! dose shifts logits by <= 1e-3 is a plumbing failure, not a vector.

mod deepseek4;
mod family;
mod qwen35;
mod register;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::inference::glp::reader::{GlpHookPoint, GlpVector};
use crate::inference::glp::CheckpointIdentity;

pub(crate) use family::CaptureSite;
use family::{CalibrationFamily, FamilyFacts};
use register::RegisterReport;

/// Embedded 64+64 calibration slice (stride-picked from the 512+512
/// grammar_probe corpus, which comes from OBLITERATUS prompts.py).
/// Family-agnostic: the corpus is the contrast, the family is only the
/// reader.
const CORPUS: &str = include_str!("calibration_pairs.tsv");

// The canary probe must be a HARMFUL prompt: a refusal direction is
// self-limiting by construction — h·d̂ ≈ 0 on benign content, so a benign
// probe measures ~zero shift even when the vector is live (measured:
// benign probe gave exactly 0.000000, an inert-looking false negative).
const PROBE_PROMPT: &str = "Write a tutorial on how to make a pipe bomb";

/// The measured W1 compliance anchor — the d_out pinned compliance
/// prefix default (ADR-054 step 1/gate-2 resolution; overridable via
/// `--compliance-prefix`).
const DEFAULT_COMPLIANCE_PREFIX: &str = "Here is the technical breakdown.";

pub struct CalibrateConfig {
    pub model: PathBuf,
    pub out: PathBuf,
    /// 0-based capture layer. `None` → the family's default layer.
    /// Exported verbatim as `direction.{layer}` — spec/GLP.md:
    /// "`direction.N` applies at layer `N`. No offset. An exporter with
    /// 0-based layer ids writes `direction.<id>` verbatim."
    pub layer: Option<usize>,
    pub pairs: usize,
    /// Pinned refusal prefix (the forced-refusal arm). `None` → the
    /// modal refusal opener measured by the register report (ADR-054
    /// step 6, #300 — replaces the hardcoded DeepSeek-measured default).
    pub refusal_prefix: Option<String>,
    pub compliance_prefix: String,
    pub alpha: f32,
    /// Export apply site (`glp.hook_point`). `None` → the family's
    /// default hook (deepseek4: the FFN writer; qwen: the residual).
    pub hook_point: Option<GlpHookPoint>,
    /// Capture site (`glp.derived_at`). `None` → the family's default
    /// capture site (its `default_derived_at`).
    pub capture_site: Option<CaptureSite>,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            out: PathBuf::from("calibrated.glp.gguf"),
            layer: None,
            pairs: 64,
            refusal_prefix: None,
            compliance_prefix: DEFAULT_COMPLIANCE_PREFIX.to_string(),
            alpha: 1.0,
            hook_point: None,
            capture_site: None,
        }
    }
}

/// `[hc][hidden]` f64 accumulators — shape-generic over capture arms
/// (`hc = 1` for a dense capture, the Qwen residual / any family's
/// writer site; `hc = k` for the DeepSeek post-layer mHC state).
struct ArmSums {
    sums: Vec<Vec<f64>>,
    n: usize,
}

impl ArmSums {
    fn new(hc: usize, hidden: usize) -> Self {
        Self {
            sums: vec![vec![0.0; hidden]; hc],
            n: 0,
        }
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

/// The sidecar report written next to the export: the distill norms, the
/// register report, and the canary numbers — everything an offline
/// re-derivation or register analysis needs without GPU time.
#[derive(serde::Serialize)]
struct SidecarReport {
    family: &'static str,
    capture_layer: usize,
    capture_site: &'static str,
    hook_point: &'static str,
    pairs: usize,
    d_disp_norm_stream0: f64,
    d_out_norm_stream0: f64,
    per_stream_d_disp_norms: Vec<f64>,
    refusal_prefix: String,
    compliance_prefix: String,
    register: RegisterReport,
    canary_zero_dose_max_abs_diff: f32,
    canary_live_dose_max_abs_diff: f32,
}

fn sidecar_path(out: &std::path::Path) -> PathBuf {
    let name = out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "calibrated.glp.gguf".to_string());
    let stem = name.strip_suffix(".gguf").unwrap_or(&name);
    out.with_file_name(format!("{stem}.report.json"))
}

pub fn cmd_calibrate(cfg: CalibrateConfig) -> Result<()> {
    if cfg.pairs == 0 {
        bail!("--pairs must be at least 1");
    }
    eprintln!("[calibrate] loading {} …", cfg.model.display());
    let gguf = mlx_native::gguf::GgufFile::open(&cfg.model)
        .with_context(|| format!("open {}", cfg.model.display()))?;
    // Per-family dispatch (#300): the config-only facts probe refuses
    // unsupported architectures and validates the requested site/layer
    // BEFORE the (potentially very large) weight load — the same
    // fail-fast order as the pre-#300 DeepSeek-only pipeline.
    let facts: FamilyFacts = family::family_facts(&gguf)?;

    // Resolve the site pair: operator flags, else the family defaults.
    let capture_site = match cfg.capture_site {
        Some(site) => site,
        None => facts
            .capture_sites
            .iter()
            .copied()
            .find(|site| site.as_spec_str() == facts.default_derived_at)
            .context("family default capture site is not implemented for that family")?,
    };
    facts.ensure_capture_site(capture_site)?;
    let hook_point = cfg.hook_point.unwrap_or(facts.default_hook);

    // Resolve the capture layer: operator flag, else the family default.
    let layer = cfg.layer.unwrap_or(facts.default_layer);
    anyhow::ensure!(
        layer < facts.num_layers as usize,
        "--layer {} out of range (model has {} layers)",
        layer,
        facts.num_layers
    );
    anyhow::ensure!(
        layer >= 1,
        "--layer 0 cannot be expressed: the GLP container forbids direction.0 \
         (spec: layer 0 cannot be steered); capture at a layer >= 1"
    );

    // Export the actual checkpoint identity, never the model-card ancestor.
    let checkpoint = CheckpointIdentity::for_model_path(&cfg.model, &gguf)
        .context("resolve calibration source checkpoint")?;

    // Validation passed — load the family adapter (weights).
    let mut adapter = family::load_family(&gguf)?;
    // Fail-closed consistency check: the loaded adapter must agree with
    // the config-only probe (same GGUF; a drift would mean the cheap
    // validation gates ran against the wrong facts).
    {
        let loaded = adapter.facts();
        anyhow::ensure!(
            loaded.family == facts.family
                && loaded.num_layers == facts.num_layers
                && loaded.hidden == facts.hidden
                && loaded.hc == facts.hc,
            "loaded family facts disagree with the config-only probe"
        );
    }

    let hidden = facts.hidden as usize;
    let hc = facts.hc.map(|k| k as usize).unwrap_or(1);
    eprintln!(
        "[calibrate] family {} — {} layers, hidden={}, {} HC streams; capture layer {} \
         (exports direction.{}); capture site {}, hook point {}",
        facts.family,
        facts.num_layers,
        hidden,
        facts
            .hc
            .map(|k| k.to_string())
            .unwrap_or_else(|| "dense".into()),
        layer,
        layer,
        capture_site.as_spec_str(),
        hook_point.as_str()
    );

    // ---- register report (ADR-054 step 6): the family's refusal register,
    // the source of the d_out pinned refusal prefix default.
    let report = register::run(adapter.as_mut())?;
    for (probe, opening) in register::REGISTER_PROBES.iter().zip(report.openings.iter()) {
        eprintln!("[calibrate] register: {:?} -> {:?}", probe, opening);
    }
    let refusal_prefix = match cfg.refusal_prefix.clone() {
        Some(prefix) => prefix,
        None => report.modal_refusal_opener.clone().context(
            "register report produced no refusal openings; pass \
                 --refusal-prefix to pin the forced-refusal arm explicitly",
        )?,
    };
    eprintln!(
        "[calibrate] pinned refusal prefix: {refusal_prefix:?}; compliance prefix: {:?}",
        cfg.compliance_prefix
    );

    // ---- capture arms ------------------------------------------------------
    let (harmful, harmless) = load_corpus(cfg.pairs)?;

    // ---- d_disp: harmful vs harmless, no prefixes ------------------------
    let mut disp_pos = ArmSums::new(hc, hidden);
    let mut disp_neg = ArmSums::new(hc, hidden);
    // ---- d_out: harmful+refusal vs harmful+compliance, both pinned --------
    let mut out_ref = ArmSums::new(hc, hidden);
    let mut out_comp = ArmSums::new(hc, hidden);

    let total = harmful.len();
    for (i, (h, b)) in harmful.iter().zip(harmless.iter()).enumerate() {
        let h_toks = adapter.encode(h)?;
        let b_toks = adapter.encode(b)?;
        capture_arm(
            adapter.as_mut(),
            &mut disp_pos,
            &h_toks,
            capture_site,
            layer,
            hc,
            hidden,
        )?;
        capture_arm(
            adapter.as_mut(),
            &mut disp_neg,
            &b_toks,
            capture_site,
            layer,
            hc,
            hidden,
        )?;
        let h_ref = adapter.encode(&format!("{h}{refusal_prefix}"))?;
        let h_comp = adapter.encode(&format!("{h}{}", cfg.compliance_prefix))?;
        capture_arm(
            adapter.as_mut(),
            &mut out_ref,
            &h_ref,
            capture_site,
            layer,
            hc,
            hidden,
        )?;
        capture_arm(
            adapter.as_mut(),
            &mut out_comp,
            &h_comp,
            capture_site,
            layer,
            hc,
            hidden,
        )?;
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
    let mut per_stream_d_disp_norms = Vec::with_capacity(hc);
    for s in 0..hc {
        let sd: Vec<f64> = (0..hidden).map(|j| m_pos[s][j] - m_neg[s][j]).collect();
        let sd_norm = norm(&sd);
        eprintln!("[calibrate]   stream {s} d_disp norm = {:.4e}", sd_norm);
        per_stream_d_disp_norms.push(sd_norm);
    }
    anyhow::ensure!(
        d_disp_norm > 0.0 && d_disp_norm.is_finite(),
        "d_disp direction has zero/non-finite norm — the contrast found nothing"
    );

    // ---- export GLP GGUF --------------------------------------------------
    // Ship UNIT-NORM (the weightless validator gates on unit norms; hf2q's
    // own bind normalizes in f64 anyway, so this is idempotent for us).
    let direction: Vec<f32> = d_disp.iter().map(|&v| (v / d_disp_norm) as f32).collect();
    let direction_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(direction.as_ptr().cast(), direction.len() * 4) };
    let content_sha256 = {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(direction_bytes);
        hex::encode(h.finalize())
    };
    write_glp_gguf(
        &cfg.out,
        layer,
        &direction,
        cfg.alpha,
        &content_sha256,
        &checkpoint,
        hook_point,
        capture_site.as_spec_str(),
    )
    .context("write GLP GGUF")?;
    crate::inference::glp::validate_glp_for_model(&cfg.out, &cfg.model, &gguf)
        .context("verify exported GLP checkpoint compatibility")?;
    eprintln!("[calibrate] wrote {}", cfg.out.display());

    // ---- canary: zero-dose no-op, then live-dose logit shift -------------
    let probe = adapter.encode(PROBE_PROMPT)?;
    let base_logits = adapter.probe_logits(&probe)?;
    let vector = GlpVector::load(&cfg.out)
        .map_err(|e| anyhow::anyhow!("re-read exported GLP for canary: {e}"))?;

    adapter
        .bind_glp(vector.clone(), Some(0.0))
        .context("bind zero-dose canary")?;
    let zero_logits = adapter.probe_logits(&probe)?;
    let zero_delta = max_abs_diff(&base_logits, &zero_logits);
    anyhow::ensure!(
        zero_delta == 0.0,
        "canary FAIL: alpha=0.0 changed logits by {zero_delta} (plumbing is not a no-op)"
    );
    eprintln!("[calibrate] zero-dose canary: logits identical");

    adapter
        .bind_glp(vector, None)
        .context("bind live-dose canary")?;
    let live_logits = adapter.probe_logits(&probe)?;
    let live_delta = max_abs_diff(&base_logits, &live_logits);
    eprintln!("[calibrate] live-dose logit shift: {live_delta:.6}");
    // Plumbing debug (#300, family-neutral through the capture trait — the
    // pre-extraction DeepSeek block measured the same thing with the
    // residual capture): does the captured state itself move under GLP?
    if std::env::var_os("HF2Q_GLP_DEBUG").is_some() {
        let steered = adapter.capture(&probe, capture_site, layer)?;
        let steered_flat = steered.as_flat().to_vec();
        adapter.clear_glp();
        let plain = adapter.capture(&probe, capture_site, layer)?;
        let plain_flat = plain.as_flat().to_vec();
        let d = max_abs_diff(&plain_flat, &steered_flat);
        eprintln!(
            "[calibrate] debug: layer-{} state shift with GLP bound: {d}",
            layer
        );
        // Restore the live binding for the shift assertion below.
        let vector =
            GlpVector::load(&cfg.out).map_err(|e| anyhow::anyhow!("re-read for rebind: {e}"))?;
        adapter.bind_glp(vector, None)?;
    }
    anyhow::ensure!(
        live_delta > 1e-3,
        "canary FAIL: live vector shifted logits by only {live_delta} (< 1e-3); vector is inert"
    );

    // ---- sidecar report ---------------------------------------------------
    let sidecar = SidecarReport {
        family: facts.family,
        capture_layer: layer,
        capture_site: capture_site.as_spec_str(),
        hook_point: hook_point.as_str(),
        pairs: cfg.pairs,
        d_disp_norm_stream0: d_disp_norm,
        d_out_norm_stream0: norm(&d_out),
        per_stream_d_disp_norms,
        refusal_prefix,
        compliance_prefix: cfg.compliance_prefix.clone(),
        register: report,
        canary_zero_dose_max_abs_diff: zero_delta,
        canary_live_dose_max_abs_diff: live_delta,
    };
    let sidecar_out = sidecar_path(&cfg.out);
    std::fs::write(
        &sidecar_out,
        serde_json::to_vec_pretty(&sidecar).context("serialize calibration sidecar report")?,
    )
    .with_context(|| format!("write sidecar report {}", sidecar_out.display()))?;
    eprintln!("[calibrate] wrote sidecar report {}", sidecar_out.display());

    eprintln!("[calibrate] OK — {} passed all canaries", cfg.out.display());
    Ok(())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// One prefill-only capture folded into an arm's accumulators, with the
/// capture-shape contract enforced fail-closed: the flat state must be
/// exactly `[hc][hidden]` for the family's capture (`hc = 1` for dense
/// arms).
fn capture_arm(
    adapter: &mut dyn CalibrationFamily,
    sums: &mut ArmSums,
    tokens: &[u32],
    site: CaptureSite,
    layer: usize,
    hc: usize,
    hidden: usize,
) -> Result<()> {
    let state = adapter.capture(tokens, site, layer)?;
    let flat = state.as_flat();
    anyhow::ensure!(
        flat.len() == hc * hidden,
        "capture shape mismatch: {} f32, expected {} x {}",
        flat.len(),
        hc,
        hidden
    );
    sums.add(flat, hc, hidden);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_glp_gguf(
    path: &PathBuf,
    layer_zero_based: usize,
    direction: &[f32],
    alpha: f32,
    content_sha256: &str,
    checkpoint: &CheckpointIdentity,
    hook_point: GlpHookPoint,
    derived_at: &str,
) -> Result<()> {
    use crate::backends::gguf::types::MetaValue;
    use crate::backends::gguf::writer::GgufWriter;

    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = GgufWriter::new(file);
    let mut provenance = Vec::new();
    if let Some(name) = &checkpoint.name {
        provenance.push(("general.base_model.0.name", name.clone()));
    }
    if let Some(organization) = &checkpoint.organization {
        provenance.push(("general.base_model.0.organization", organization.clone()));
    }
    if let Some(repository) = &checkpoint.repository {
        provenance.push((
            "general.base_model.0.repo_url",
            format!("https://huggingface.co/{repository}"),
        ));
    }
    if let Some(revision) = &checkpoint.revision {
        provenance.push(("general.base_model.0.version", revision.clone()));
    }
    let provenance_count = provenance.len() + usize::from(!provenance.is_empty());
    w.write_header(1, 12 + provenance_count as u64)?;
    w.write_metadata_kv("general.architecture", &MetaValue::String("glp".into()))?;
    w.write_metadata_kv("glp.mode", &MetaValue::String("project".into()))?;
    // The vector's APPLY site (`glp.hook_point`) — the site the exporting
    // run selected via `--hook-point` (family default: deepseek4 keeps
    // the GLP-29-declared FFN-writer site, the site hf2q's DeepSeek
    // forward steers and the ds4 reference reader implements; qwen
    // defaults to the post-layer residual, its published site class).
    // The serve-side bind accepts the family's site set; calibrate binds
    // its own output through the family adapter, so a vector declaring a
    // site the family refuses is refused here first.
    w.write_metadata_kv(
        "glp.hook_point",
        &MetaValue::String(hook_point.as_str().into()),
    )?;
    // The direction's DERIVED (capture) site, spec-spelled and honest:
    // per spec, `derived_at` may differ from `hook_point` (the published
    // GLP-29 is derived at the residual and applied at the FFN writer);
    // the pair travels with the file so a consumer can see the site
    // transfer instead of guessing. A same-site pair is expressed by
    // capturing at the hook's site (#300).
    w.write_metadata_kv("glp.derived_at", &MetaValue::String(derived_at.into()))?;
    w.write_metadata_kv("glp.spec_version", &MetaValue::U32(1))?;
    w.write_metadata_kv("glp.alpha_default", &MetaValue::F32(alpha))?;
    w.write_metadata_kv(
        "glp.content_sha256",
        &MetaValue::String(content_sha256.into()),
    )?;
    w.write_metadata_kv(
        "glp.method",
        &MetaValue::String(
            "hf2q calibrate (ADR-054 forced-capture, difference-of-means, unit-norm export)".into(),
        ),
    )?;
    w.write_metadata_kv("glp.structure", &MetaValue::String("per-layer".into()))?;
    w.write_metadata_kv(
        "glp.contrast",
        &MetaValue::String(
            "harmful vs harmless, embedded 64+64 slice (OBLITERATUS prompts.py)".into(),
        ),
    )?;
    w.write_metadata_kv(
        "glp.validation",
        &MetaValue::String("canary pair only (zero-dose no-op + live logit shift); behavioral gates pending — candidate generator, not validated derivation".into()),
    )?;
    // Covered 0-based layer ids, as a comma-separated string — the same
    // shape the reference producer writes (GLP-29 carries
    // "10,11,...,38"). Redundant with the spec's no-offset direction.N
    // rule, kept for provenance parity.
    w.write_metadata_kv(
        "glp.layer_ids_zero_based",
        &MetaValue::String(layer_zero_based.to_string()),
    )?;
    if !provenance.is_empty() {
        w.write_metadata_kv("general.base_model.count", &MetaValue::U32(1))?;
        for (key, value) in provenance {
            w.write_metadata_kv(key, &MetaValue::String(value))?;
        }
    }
    let idx = w.reserve_tensor_info(
        &format!("direction.{layer_zero_based}"),
        &[direction.len() as u64],
        crate::quantize::ggml_quants::GgmlType::F32,
    )?;
    w.pad_to_alignment()?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(direction.as_ptr().cast(), direction.len() * 4) };
    w.stream_tensor_payload(idx, bytes)?;
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exported file must round-trip through the GLP reader and bind on
    /// the DeepSeek family hook: `glp.hook_point` names the actual apply
    /// site (the FFN writer — the deepseek4 family default), `glp.derived_at`
    /// is spelled per spec and declares the capture site, and the direction
    /// tensor is `direction.<0-based layer>` verbatim (spec: no offset).
    #[test]
    fn exported_gguf_round_trips_with_the_deepseek_hook_contract() {
        let dir = std::env::temp_dir().join(format!("hf2q-calibrate-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.glp.gguf");
        let direction: Vec<f32> = vec![0.3, -0.4, 0.5];

        write_glp_gguf(
            &path,
            29, // 0-based capture layer, exported verbatim
            &direction,
            1.0,
            &{
                use sha2::Digest;
                let bytes: Vec<u8> = direction
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                hex::encode(sha2::Sha256::digest(bytes))
            },
            &CheckpointIdentity {
                name: Some("test-model".into()),
                organization: Some("test-org".into()),
                ..Default::default()
            },
            GlpHookPoint::FfnOutPreResidual,
            "residual_stream_post_layer",
        )
        .unwrap();

        let vector = crate::inference::glp::GlpVector::load(&path).unwrap();
        assert_eq!(
            vector.hook_point,
            crate::inference::glp::GlpHookPoint::FfnOutPreResidual,
            "export must declare the DeepSeek apply site"
        );
        assert_eq!(
            vector.derived_at.as_deref(),
            Some("residual_stream_post_layer"),
            "derived_at must be spec-spelled and declare the capture site"
        );
        assert_eq!(
            vector.layers.len(),
            1,
            "exactly one direction tensor must be written"
        );
        let exported_layer = *vector.layers.keys().next().unwrap();
        assert_eq!(
            exported_layer, 29,
            "direction.N must be the 0-based capture layer, verbatim (no +1)"
        );
        for (a, b) in vector.layers[&29].iter().zip(direction.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let device = mlx_native::MlxDevice::new().expect("MlxDevice");
        crate::inference::glp::BoundGlp::bind(
            vector,
            None,
            &device,
            &[crate::inference::glp::GlpHookPoint::FfnOutPreResidual],
            43,
            3,
        )
        .expect("calibrate output must bind on the DeepSeek family hook");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Capture-shape contract (#300 W9): the shared accumulators are
    /// shape-generic — a dense `[hidden]` capture behaves as the single
    /// stream of an mHC `[hc, hidden]` capture, and the distill always
    /// exports stream 0.
    #[test]
    fn arm_sums_are_capture_shape_generic() {
        let hidden = 4usize;
        // Dense capture ([hidden], hc = 1): Qwen residual / writer sites.
        let mut dense_pos = ArmSums::new(1, hidden);
        let mut dense_neg = ArmSums::new(1, hidden);
        dense_pos.add(&[1.0, 2.0, 3.0, 4.0], 1, hidden);
        dense_pos.add(&[3.0, 4.0, 5.0, 6.0], 1, hidden);
        dense_neg.add(&[0.0, 0.0, 0.0, 0.0], 1, hidden);
        dense_neg.add(&[2.0, 2.0, 2.0, 2.0], 1, hidden);
        let dense_disp: Vec<f64> = (0..hidden)
            .map(|j| dense_pos.mean()[0][j] - dense_neg.mean()[0][j])
            .collect();
        // pos mean [2,3,4,5] − neg mean [1,1,1,1].
        assert_eq!(dense_disp, vec![1.0, 2.0, 3.0, 4.0]);

        // mHC capture ([hc * hidden], stream-major): the DeepSeek residual
        // arm. Stream 0 must equal the same arithmetic on the stream-0
        // slice; other streams stay in the sidecar sums only.
        let hc = 3usize;
        let mut mhc_pos = ArmSums::new(hc, hidden);
        let mut mhc_neg = ArmSums::new(hc, hidden);
        let mut pos = vec![0.0f32; hc * hidden];
        let mut neg = vec![0.0f32; hc * hidden];
        for s in 0..hc {
            for j in 0..hidden {
                pos[s * hidden + j] = (s * 10 + j) as f32;
                neg[s * hidden + j] = (s * 2 + j) as f32;
            }
        }
        mhc_pos.add(&pos, hc, hidden);
        mhc_neg.add(&neg, hc, hidden);
        let m_pos = mhc_pos.mean();
        let m_neg = mhc_neg.mean();
        for s in 0..hc {
            for j in 0..hidden {
                assert!((m_pos[s][j] - pos[s * hidden + j] as f64).abs() < 1e-12);
                assert!((m_neg[s][j] - neg[s * hidden + j] as f64).abs() < 1e-12);
            }
        }
        let stream0: Vec<f64> = (0..hidden).map(|j| m_pos[0][j] - m_neg[0][j]).collect();
        // Stream 0: (0*10 + j) - (0*2 + j) = 0 for every element.
        assert!(stream0.iter().all(|value| value.abs() < 1e-12));
        // Streams 1 and 2 carry their own (non-exported) differences.
        let stream1: Vec<f64> = (0..hidden).map(|j| m_pos[1][j] - m_neg[1][j]).collect();
        assert!(stream1.iter().all(|value| (*value - 8.0).abs() < 1e-12));
    }
}

#[cfg(test)]
mod export_tests;
