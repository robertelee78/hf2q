//! ADR-063 Q4-GEMMA-SPEC-SERVE M0 (design §8.1): the round-cost curve and
//! gate G0, measured on the R-A verify route with forced drafts, target
//! only — no drafter runtime.
//!
//! Case file (`HF2Q_GEMMA_SPEC_M0_CASE`, JSON):
//! ```json
//! { "model": "/path/target.gguf", "context": 32768,
//!   "lanes": [[token ids …] | {"request": "/…/turn-N/request.json",
//!             "prompt_tokens": 1222}, …4 lanes],
//!   "tick_seconds_by_width": {"1": s, "2": s, "3": s, "4": s},
//!   "output": "/path/evidence/dir", "warmup": 20, "samples": 200 }
//! ```
//! - `lanes`: rendered prompt token ids, either literal or a recorded
//!   chat-completions request rendered with the loaded model's template and
//!   tokenizer exactly as the server does (tools in scope, template-default
//!   thinking, no kwargs); the rendered length must equal the recorded
//!   `prompt_tokens`. Each lane is prefilled except its last four tokens;
//!   those four are the forced window `[x, d0, d1, d2]` at `pos0 = len − 4`.
//!   Lane 0 and at least one other lane must reach the wrapped ring.
//! - `tick_seconds_by_width`: `hf2q_gemma_decode_tick_seconds_total{width,
//!   speculative="false"}` from one spec-off Q5 run of the same binary.
//!
//! Every timed tick runs at the same positions without committing a cursor:
//! the ordinary tick rewrites its own row, and each verify tick saves the
//! ring pre-image and restores it in full afterwards, so the state is
//! stationary. Before timing, a Class A check compares every verify row
//! bitwise to the teacher-forced ordinary ticks at the same positions (at
//! widths 1 and 4, both head paths) and re-runs the reference after the
//! verifies to prove the restore returned the readable state.
//!
//! Output: `<output>/m0-round-cost.json`.

use std::collections::BTreeMap;
use std::time::Instant;

use super::*;
use crate::inference::models::gemma4::batched_body::BatchedKvRegime;
use crate::inference::models::gemma4::spec_cost::{
    decide_g0, median, DrafterBoundParts, RoundCostMeasurements,
};
use crate::inference::models::gemma4::spec_verify::{
    sliding_layer_geometry, DecodeWindow, GemmaRingPreimage, RingPreimagePlan, WindowRows,
    MAX_WINDOW,
};

const CASE_ENV: &str = "HF2Q_GEMMA_SPEC_M0_CASE";
const LANES: usize = 4;
const Q8_0_BLOCK_BYTES: usize = 34;
const DRAFTER_HEADS: usize = 16;
const DRAFTER_VOCAB: usize = 262_144;
const EMPTY_DISPATCHES: usize = 64;
const MV_COPY_BUDGET_BYTES: usize = 256 << 20;

#[derive(serde::Deserialize)]
struct M0Case {
    model: PathBuf,
    context: u32,
    lanes: Vec<LaneInput>,
    tick_seconds_by_width: BTreeMap<String, f64>,
    output: PathBuf,
    #[serde(default = "default_warmup")]
    warmup: usize,
    #[serde(default = "default_samples")]
    samples: usize,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum LaneInput {
    Tokens(Vec<u32>),
    Request {
        request: PathBuf,
        prompt_tokens: usize,
    },
}

impl LaneInput {
    fn tokens(&self, model: &GemmaLoadedModel) -> Result<Vec<u32>> {
        let (request, expected) = match self {
            Self::Tokens(tokens) => return Ok(tokens.clone()),
            Self::Request {
                request,
                prompt_tokens,
            } => (request, *prompt_tokens),
        };
        let wire = std::fs::read(request).with_context(|| format!("read {request:?}"))?;
        let req: super::super::schema::ChatCompletionRequest = serde_json::from_slice(&wire)?;
        anyhow::ensure!(
            req.chat_template_kwargs.is_none(),
            "M0 lane request {request:?} carries chat_template_kwargs"
        );
        let template = model.chat_template.as_str();
        let thinking = crate::serve::template_supports_enable_thinking(template);
        let rendered = render_chat_prompt_with_tools_generation_prompt(
            template,
            &req.messages,
            req.tools.as_deref(),
            thinking,
            None,
            true,
        )?;
        let tokens = model
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| anyhow::anyhow!("tokenize {request:?}: {e}"))?
            .get_ids()
            .to_vec();
        anyhow::ensure!(
            tokens.len() == expected,
            "M0 lane {request:?} renders {} tokens, recorded prompt_tokens {expected}",
            tokens.len()
        );
        Ok(tokens)
    }

    fn describe(&self, tokens: &[u32]) -> serde_json::Value {
        let ids = tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>();
        let digest = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&ids));
        match self {
            Self::Tokens(_) => serde_json::json!({"source": "token_ids", "tokens": tokens.len(),
                "token_ids_le_sha256": digest}),
            Self::Request { request, .. } => serde_json::json!({"source": "request",
                "request": request, "request_sha256": std::fs::read(request).ok()
                    .map(|b| hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&b))),
                "tokens": tokens.len(), "token_ids_le_sha256": digest}),
        }
    }
}

fn default_warmup() -> usize {
    20
}

fn default_samples() -> usize {
    200
}

struct Lane {
    slot: SlotId,
    pos0: usize,
    tail: [u32; MAX_WINDOW],
}

struct Harness {
    model: GemmaLoadedModel,
    lanes: Vec<Lane>,
    preimages: Vec<GemmaRingPreimage>,
    ring_capacity: usize,
    vocab: usize,
}

impl Harness {
    fn windows(&self, n: usize, w: usize, offset: usize) -> Vec<DecodeWindow> {
        self.lanes[..n]
            .iter()
            .map(|lane| DecodeWindow {
                slot: lane.slot,
                start_pos: lane.pos0 + offset,
                tokens: lane.tail[offset..offset + w].to_vec(),
            })
            .collect()
    }

    /// One decode forward (body + head) over `windows`; logits in row order.
    fn forward(&mut self, windows: &[DecodeWindow]) -> Result<(Vec<f32>, WindowRows)> {
        let mut guard = Gemma4KvGuard::take(&mut self.model)?;
        clear_gemma4_self_mounts(guard.model);
        let mut fused = None;
        let lm = &mut *guard.model;
        let mut regime =
            BatchedKvRegime::Hybrid(guard.hybrid.as_mut().context("Gemma hybrid scaffold")?);
        let (hidden, rows) =
            lm.weights
                .forward_decode_body_batched(windows, &mut regime, &mut fused, &mut lm.ctx)?;
        let head = match fused {
            Some(head) => head,
            None => lm.weights.lm_head_batched(&hidden, rows.rows(), &mut lm.ctx)?,
        };
        anyhow::ensure!(
            head.logits.len() == rows.rows() * self.vocab,
            "Gemma M0 logits extent {} != {} rows",
            head.logits.len(),
            rows.rows()
        );
        Ok((head.logits, rows))
    }

    /// The host walk of a forced-draft round: argmax each lane's rows in
    /// order until the first disagreement. Returns emitted rows.
    fn walk(&self, logits: &[f32], rows: &WindowRows, windows: &[DecodeWindow]) -> usize {
        let mut emitted = 0;
        for (index, window) in windows.iter().enumerate() {
            for (rank, &row) in rows.row_of[index].iter().enumerate() {
                let (id, _) = argmax_f32_first_max(&logits[row * self.vocab..(row + 1) * self.vocab]);
                emitted += 1;
                if window.tokens.get(rank + 1).is_some_and(|&next| next != id) {
                    break;
                }
            }
        }
        emitted
    }

    fn plans(&self, n: usize, k: usize) -> Result<Vec<(usize, RingPreimagePlan)>> {
        self.lanes[..n]
            .iter()
            .enumerate()
            .map(|(index, lane)| Ok((index, RingPreimagePlan::new(lane.pos0, k, self.ring_capacity)?)))
            .collect()
    }

    /// Encode and wait one save (`restore_m = None`) or restore after `m`
    /// committed rows over every lane that needs it. Returns whether any
    /// dispatch ran.
    fn ring(&mut self, plans: &[(usize, RingPreimagePlan)], restore_m: Option<usize>) -> Result<bool> {
        let guard = Gemma4KvGuard::take(&mut self.model)?;
        let hybrid = guard.hybrid.as_ref().context("Gemma hybrid scaffold")?;
        let (exec, reg) = guard.model.ctx.split();
        let dev = exec.device();
        let mut session = crate::inference::native_observation::session(
            exec.begin().map_err(|e| anyhow::anyhow!("M0 ring session: {e}"))?,
        );
        let mut any = false;
        for (index, plan) in plans {
            let slot = self.lanes[*index].slot;
            match restore_m {
                None => {
                    if let Some(save) = plan.save {
                        self.preimages[*index].encode_save(&mut session, reg, dev, hybrid, slot, &save)?;
                        any = true;
                    }
                }
                Some(m) => {
                    if let Some(restore) = plan.restore(m)? {
                        self.preimages[*index]
                            .encode_restore(&mut session, reg, dev, hybrid, slot, &restore)?;
                        any = true;
                    }
                }
            }
        }
        session
            .finish()
            .map_err(|e| anyhow::anyhow!("M0 ring finish: {e}"))?;
        Ok(any)
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Class A on this artifact and these lanes: each verify row equals the
/// teacher-forced ordinary tick at its position, and the restore returns
/// the state the reference reads.
fn class_a_check(h: &mut Harness, n: usize) -> Result<serde_json::Value> {
    let vocab = h.vocab;
    let all = h.plans(n, MAX_WINDOW - 1)?;
    let reference = |h: &mut Harness| -> Result<Vec<Vec<u32>>> {
        h.ring(&all, None)?;
        let mut out = Vec::new();
        for offset in 0..MAX_WINDOW {
            let (logits, _) = h.forward(&h.windows(n, 1, offset))?;
            out.push(bits(&logits));
        }
        h.ring(&all, Some(1))?;
        Ok(out)
    };
    let first = reference(h)?;
    let mut rows_checked = 0usize;
    for w in 2..=MAX_WINDOW {
        let plans = h.plans(n, w - 1)?;
        h.ring(&plans, None)?;
        let windows = h.windows(n, w, 0);
        let (logits, rows) = h.forward(&windows)?;
        h.ring(&plans, Some(1))?;
        for lane in 0..n {
            for rank in 0..w {
                let row = rows.row_of[lane][rank];
                let got = bits(&logits[row * vocab..(row + 1) * vocab]);
                let want = &first[rank][lane * vocab..(lane + 1) * vocab];
                if got != want {
                    let at = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(0);
                    anyhow::bail!(
                        "Class A FALSIFIED: w={w} n={n} lane={lane} rank={rank} \
                         pos={} first differing logit {at}",
                        h.lanes[lane].pos0 + rank
                    );
                }
                rows_checked += 1;
            }
        }
    }
    let second = reference(h)?;
    anyhow::ensure!(
        first == second,
        "Class A FALSIFIED: teacher-forced reference changed after verify + restore (n={n})"
    );
    Ok(serde_json::json!({"width": n, "verify_rows_bitwise_equal": rows_checked,
        "reference_rerun_bitwise_equal": true}))
}

fn timed<F: FnMut() -> Result<()>>(warmup: usize, samples: usize, mut f: F) -> Result<f64> {
    for _ in 0..warmup {
        f()?;
    }
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        f()?;
        out.push(start.elapsed().as_secs_f64());
    }
    median(out)
}

fn gpu_timed<F: FnMut(&mut mlx_native::graph::GraphSession<'_>, &mut mlx_native::KernelRegistry, &mlx_native::MlxDevice, usize) -> Result<()>>(
    gpu: &mut crate::serve::gpu::GpuContext,
    warmup: usize,
    samples: usize,
    mut encode: F,
) -> Result<f64> {
    let mut out = Vec::with_capacity(samples);
    for iteration in 0..warmup + samples {
        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let mut session = exec.begin().map_err(|e| anyhow::anyhow!("M0 bench session: {e}"))?;
        encode(&mut session, reg, dev, iteration)?;
        let ns = session
            .finish_with_gpu_time()
            .map_err(|e| anyhow::anyhow!("M0 bench finish: {e}"))?;
        if iteration >= warmup {
            out.push(ns as f64 * 1e-9);
        }
    }
    median(out)
}

fn random_q8_0(dev: &mlx_native::MlxDevice, rows: usize, cols: usize, seed: &mut u64) -> Result<mlx_native::MlxBuffer> {
    anyhow::ensure!(cols % 32 == 0, "Q8_0 columns must be a multiple of 32");
    let bytes = rows * (cols / 32) * Q8_0_BLOCK_BYTES;
    let mut buffer = dev
        .alloc_buffer(bytes, mlx_native::DType::U8, vec![bytes])
        .map_err(|e| anyhow::anyhow!("Q8_0 weight alloc: {e}"))?;
    let scale = half::f16::from_f32(0.01).to_bits().to_le_bytes();
    for block in buffer
        .as_mut_slice::<u8>()
        .map_err(|e| anyhow::anyhow!("Q8_0 weight write: {e}"))?
        .chunks_exact_mut(Q8_0_BLOCK_BYTES)
    {
        block[..2].copy_from_slice(&scale);
        for byte in &mut block[2..] {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *byte = *seed as u8;
        }
    }
    Ok(buffer)
}

fn random_f32(dev: &mlx_native::MlxDevice, elements: usize, seed: &mut u64) -> Result<mlx_native::MlxBuffer> {
    let mut buffer = dev
        .alloc_buffer(elements * 4, mlx_native::DType::F32, vec![elements])
        .map_err(|e| anyhow::anyhow!("F32 alloc: {e}"))?;
    for value in buffer
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("F32 write: {e}"))?
    {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *value = ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    }
    Ok(buffer)
}

fn u32_buffer(dev: &mlx_native::MlxDevice, values: &[u32]) -> Result<mlx_native::MlxBuffer> {
    let mut buffer = dev
        .alloc_buffer(values.len().max(1) * 4, mlx_native::DType::U32, vec![values.len().max(1)])
        .map_err(|e| anyhow::anyhow!("U32 alloc: {e}"))?;
    buffer.as_mut_slice::<u32>().map_err(|e| anyhow::anyhow!("U32 write: {e}"))?[..values.len()]
        .copy_from_slice(values);
    Ok(buffer)
}

/// §3.1 drafter mat-vec shapes `(name, out, in)` in step order.
fn drafter_shapes() -> Vec<(String, usize, usize)> {
    let mut shapes = vec![("pre_projection".to_string(), 1024, 5632)];
    for layer in 0..4 {
        let hd = if layer == 3 { 512 } else { 256 };
        let q = DRAFTER_HEADS * hd;
        shapes.push((format!("l{layer}.q"), q, 1024));
        shapes.push((format!("l{layer}.o"), 1024, q));
        shapes.push((format!("l{layer}.gate"), 8192, 1024));
        shapes.push((format!("l{layer}.up"), 8192, 1024));
        shapes.push((format!("l{layer}.down"), 1024, 8192));
    }
    shapes.push(("head".to_string(), DRAFTER_VOCAB, 1024));
    shapes.push(("post_projection".to_string(), 2816, 1024));
    shapes
}

/// Drafter-bound components at widths 1..=LANES (all existing ops).
fn drafter_bound(h: &mut Harness, warmup: usize, samples: usize) -> Result<(BTreeMap<usize, DrafterBoundParts>, serde_json::Value)> {
    let mut seed = 0x4d30_5444_424f_554eu64;
    let dev = h.model.ctx.executor.device().clone();
    let mut per_shape = serde_json::Map::new();
    let mut sum_mv: BTreeMap<usize, f64> = BTreeMap::new();
    for (name, out, inp) in drafter_shapes() {
        let bytes = out * (inp / 32) * Q8_0_BLOCK_BYTES;
        let copies = (MV_COPY_BUDGET_BYTES / bytes).clamp(1, 8);
        let weights = (0..copies)
            .map(|_| random_q8_0(&dev, out, inp, &mut seed))
            .collect::<Result<Vec<_>>>()?;
        let mut by_n = serde_json::Map::new();
        for n in 1..=LANES {
            let input = random_f32(&dev, n * inp, &mut seed)?;
            let output = dev
                .alloc_buffer(n * out * 4, mlx_native::DType::F32, vec![n * out])
                .map_err(|e| anyhow::anyhow!("mv output alloc: {e}"))?;
            let params = mlx_native::GgmlQuantizedMatmulParams {
                m: n as u32,
                n: out as u32,
                k: inp as u32,
                ggml_type: mlx_native::GgmlType::Q8_0,
            };
            let t = gpu_timed(&mut h.model.ctx, warmup, samples, |s, reg, dev, i| {
                s.quantized_matmul_ggml(reg, dev, &input, &weights[i % copies], &output, &params)
                    .map_err(|e| anyhow::anyhow!("Q8_0 mv {name} m={n}: {e}"))
            })?;
            *sum_mv.entry(n).or_default() += t;
            by_n.insert(n.to_string(), t.into());
        }
        per_shape.insert(name, by_n.into());
    }

    // Empty-dispatch cost: one-element argmax dispatches.
    let one = random_f32(&dev, 1, &mut seed)?;
    let index = u32_buffer(&dev, &[0])?;
    let value = random_f32(&dev, 1, &mut seed)?;
    let params_one = u32_buffer(&dev, &[1])?;
    let t_disp = gpu_timed(&mut h.model.ctx, warmup, samples, |s, reg, dev, _| {
        for _ in 0..EMPTY_DISPATCHES {
            s.barrier_between(&[&one], &[&index, &value]);
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(), reg, dev.metal_device(), &one, &index, &value, &params_one, 1,
            )
            .map_err(|e| anyhow::anyhow!("empty argmax: {e}"))?;
        }
        Ok(())
    })? / EMPTY_DISPATCHES as f64;

    let layers = h.model.weights.layers.len();
    anyhow::ensure!(layers >= 2, "target has fewer than two layers");
    let tq_bits = crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits();
    let tq_scale: f32 = match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
        Ok("sqrt256") => 16.0,
        Ok("sqrt512") => 512.0_f32.sqrt(),
        _ => 1.0,
    };
    let sliding_window = h.model.weights.sliding_window;
    let mut parts = BTreeMap::new();
    let mut flash_json = serde_json::Map::new();
    for n in 1..=LANES {
        let logits = random_f32(&dev, n * DRAFTER_VOCAB, &mut seed)?;
        let idx = u32_buffer(&dev, &vec![0; n])?;
        let val = random_f32(&dev, n, &mut seed)?;
        let params = u32_buffer(&dev, &[DRAFTER_VOCAB as u32])?;
        let t_argmax = gpu_timed(&mut h.model.ctx, warmup, samples, |s, reg, dev, _| {
            for row in 0..n {
                let input = logits.slice_view((row * DRAFTER_VOCAB * 4) as u64, DRAFTER_VOCAB);
                let out_i = idx.slice_view((row * 4) as u64, 1);
                let out_v = val.slice_view((row * 4) as u64, 1);
                s.barrier_between(&[&input], &[&out_i, &out_v]);
                mlx_native::ops::argmax::dispatch_argmax_f32(
                    s.encoder_mut(), reg, dev.metal_device(), &input, &out_i, &out_v, &params,
                    DRAFTER_VOCAB as u32,
                )
                .map_err(|e| anyhow::anyhow!("argmax n={n}: {e}"))?;
            }
            Ok(())
        })?;

        // Drafter attention over the live target layers n_layer−2 / n_layer−1
        // at seq_pos = pos0 − 1 (sliding window 1023 on the sliding layer).
        let mut flash_sum = 0.0;
        for layer_idx in [layers - 2, layers - 1] {
            let geometry = &h.model.weights.layers[layer_idx];
            let (hd, nkv) = (geometry.head_dim, geometry.num_kv_heads);
            let is_sliding =
                geometry.layer_type == crate::serve::config::LayerType::Sliding;
            let q = random_f32(&dev, n * DRAFTER_HEADS * hd, &mut seed)?;
            let out = random_f32(&dev, n * DRAFTER_HEADS * hd, &mut seed)?;
            let tmp = random_f32(&dev, n * (DRAFTER_HEADS * hd * 32 + DRAFTER_HEADS * 2 * 32), &mut seed)?;
            let slots: Vec<u32> = h.lanes[..n].iter().map(|l| l.slot.0).collect();
            let positions: Vec<u32> = h.lanes[..n].iter().map(|l| (l.pos0 - 1) as u32).collect();
            let slot_buf = u32_buffer(&dev, &slots)?;
            let pos_buf = u32_buffer(&dev, &positions)?;
            let guard = Gemma4KvGuard::take(&mut h.model)?;
            let kv = &guard.hybrid.as_ref().context("Gemma hybrid scaffold")?[layer_idx];
            let (cap, ring) = (kv.capacity, kv.is_sliding);
            let max_ksl = h.lanes[..n]
                .iter()
                .map(|l| if ring { l.pos0.min(cap) } else { l.pos0 } as u32)
                .max()
                .unwrap_or(1);
            let p = mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                num_heads: DRAFTER_HEADS as u32,
                num_kv_heads: nkv as u32,
                head_dim: hd as u32,
                kv_seq_len: max_ksl,
                kv_capacity: cap as u32,
                scale: 1.0,
                mask_type: if is_sliding { 2 } else { 1 },
                sliding_window: if is_sliding { (sliding_window - 1) as u32 } else { 0 },
                softcap: 0.0,
                ring_start: 0,
                scale_factor_d512: tq_scale,
                codebook_bits: tq_bits,
                fuse_fwht_pre: 0,
                nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(max_ksl),
            };
            let (k, v, norms) = (kv.k.clone(), kv.v_packed.clone(), kv.v_norms.clone());
            let t = gpu_timed(&mut guard.model.ctx, warmup, samples, |s, reg, dev, _| {
                s.barrier_between(&[&q, &k, &v, &norms, &tmp], &[&out, &tmp]);
                mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid_batched(
                    s.encoder_mut(), reg, dev, n as u32, &q, &k, &v, &norms, &out, &tmp,
                    &slot_buf, &pos_buf, &p,
                )
                .map_err(|e| anyhow::anyhow!("drafter flash L{layer_idx} n={n}: {e}"))?;
                s.barrier_between(&[&out], &[&out]);
                mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
                    s.encoder_mut(), reg, dev.metal_device(), &out,
                    (n * DRAFTER_HEADS) as u32, hd as u32,
                )
                .map_err(|e| anyhow::anyhow!("drafter undo L{layer_idx} n={n}: {e}"))
            })?;
            drop(guard);
            flash_json.insert(format!("L{layer_idx}.n{n}"), t.into());
            flash_sum += t;
        }
        parts.insert(
            n,
            DrafterBoundParts {
                sum_t_mv: sum_mv[&n],
                t_argmax,
                t_flash: flash_sum / 2.0,
                t_disp,
            },
        );
    }
    Ok((
        parts,
        serde_json::json!({"t_mv": per_shape, "t_flash": flash_json, "t_disp": t_disp}),
    ))
}

fn run() -> Result<()> {
    let path = std::env::var_os(CASE_ENV).with_context(|| format!("{CASE_ENV} required"))?;
    let raw = std::fs::read(&path).with_context(|| format!("read {path:?}"))?;
    let case: M0Case = serde_json::from_slice(&raw)?;
    anyhow::ensure!(case.lanes.len() == LANES, "M0 needs {LANES} lanes");
    anyhow::ensure!(case.samples > 0, "M0 samples must be positive");
    anyhow::ensure!(
        std::env::var("HF2Q_BATCHED_BODY").as_deref() != Ok("0")
            && crate::debug::INVESTIGATION_ENV.hybrid_kv,
        "M0 measures the production batched body on the hybrid KV scaffold"
    );
    let tick_seconds = case
        .tick_seconds_by_width
        .iter()
        .map(|(k, v)| Ok((k.parse::<usize>().with_context(|| format!("width {k}"))?, *v)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    anyhow::ensure!(
        tick_seconds.keys().all(|n| (1..=LANES).contains(n)),
        "M0 measures widths 1..={LANES}; the histogram has others"
    );
    std::fs::create_dir_all(&case.output)?;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let options = LoadOptions {
        model_path: case.model.clone(),
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
    };
    let LoadedModel::Gemma(mut model) = LoadedModel::load_with_context(&options, Some(case.context))?
    else {
        anyhow::bail!("M0 requires a Gemma 4 target")
    };
    model.provision_multi_seq_kv_for_slot_aware(LANES as u32)?;
    anyhow::ensure!(model.multi_seq_kv_hybrid.is_some(), "hybrid scaffold required");
    anyhow::ensure!(model.weights.kv_graft_len() == 0, "M0 refuses a bound KV graft");

    let mut lanes = Vec::with_capacity(LANES);
    let rendered = case
        .lanes
        .iter()
        .map(|lane| lane.tokens(&model))
        .collect::<Result<Vec<_>>>()?;
    let lane_sources: Vec<serde_json::Value> = case
        .lanes
        .iter()
        .zip(&rendered)
        .map(|(lane, tokens)| lane.describe(tokens))
        .collect();
    for (index, tokens) in rendered.iter().enumerate() {
        anyhow::ensure!(tokens.len() > MAX_WINDOW, "lane {index} shorter than its window");
        anyhow::ensure!(tokens.len() < case.context as usize, "lane {index} exceeds context");
        let pos0 = tokens.len() - MAX_WINDOW;
        let slot = SlotId(index as u32);
        let hb = model.multi_seq_kv.as_mut().context("Gemma slot scaffold")?;
        model.weights.forward_prefill_with_soft_tokens_slot_aware(
            &tokens[..pos0], &[], 8192, &mut model.ctx, slot, hb,
            model.multi_seq_kv_hybrid.as_mut(), model.multi_seq_kv_dense.as_mut(),
            model.multi_seq_kv_mlx.as_mut(),
        )?;
        q1_gemma_cohort::commit_after_checked_return(&mut model, slot, 0, pos0)?;
        let mut tail = [0u32; MAX_WINDOW];
        tail.copy_from_slice(&tokens[pos0..]);
        lanes.push(Lane { slot, pos0, tail });
    }
    let hybrid = model.multi_seq_kv_hybrid.as_ref().context("hybrid scaffold")?;
    let layers = &model.weights.layers;
    let geometry = sliding_layer_geometry(hybrid, |layer| layers[layer].num_kv_heads)?;
    let ring_capacity = geometry
        .first()
        .map(|g| g.capacity)
        .context("target has no sliding layer")?;
    let dev = model.ctx.executor.device().clone();
    let preimages = (0..LANES)
        .map(|_| GemmaRingPreimage::new(&dev, geometry.clone()))
        .collect::<Result<Vec<_>>>()?;
    let vocab = model.weights.vocab_size;
    let mut h = Harness {
        model,
        lanes,
        preimages,
        ring_capacity,
        vocab,
    };

    // The phase ordering and the pre-image only matter once the window
    // reaches the wrapped ring (`pos0 + 1 ≥ cap`): the width-1 check runs on
    // lane 0 and the width-4 check must include another wrapped lane, so a
    // case whose lanes stay below the ring would check nothing that matters.
    let wraps = |lane: &Lane| lane.pos0 + 1 >= h.ring_capacity;
    let wrapped: Vec<bool> = h.lanes.iter().map(wraps).collect();
    anyhow::ensure!(
        wrapped[0] && wrapped[1..].iter().any(|&w| w),
        "M0 case must wrap the {}-row ring in lane 0 and at least one other lane \
         (pos0 + 1 ≥ cap); lane pos0 = {:?}",
        h.ring_capacity,
        h.lanes.iter().map(|l| l.pos0).collect::<Vec<_>>()
    );
    let mut class_a = Vec::new();
    let mut measurements: BTreeMap<&str, RoundCostMeasurements> = BTreeMap::new();
    let mut flat = Vec::new();
    for (head, fuse) in [("fused", "1"), ("unfused", "0")] {
        std::env::set_var("HF2Q_FUSE_LMHEAD", fuse);
        for n in [1, LANES] {
            let mut record = class_a_check(&mut h, n)?;
            record["head"] = head.into();
            record["wrapped_lanes"] = wrapped[..n].iter().filter(|&&w| w).count().into();
            class_a.push(record);
        }
        let m = measurements.entry(head).or_default();
        for n in 1..=LANES {
            let windows = h.windows(n, 1, 0);
            let t1 = timed(case.warmup, case.samples, || {
                let (logits, rows) = h.forward(&windows)?;
                std::hint::black_box(h.walk(&logits, &rows, &windows));
                Ok(())
            })?;
            m.t1.insert(n, t1);
            flat.push(serde_json::json!({"head": head, "figure": "t1", "n": n, "seconds": t1}));
        }
        for w in 2..=MAX_WINDOW {
            let k = w - 1;
            for n in 1..=LANES {
                let plans = h.plans(n, k)?;
                let windows = h.windows(n, w, 0);
                let (mut ts, mut tv, mut tr) = (Vec::new(), Vec::new(), Vec::new());
                for iteration in 0..case.warmup + case.samples {
                    let start = Instant::now();
                    h.ring(&plans, None)?;
                    let save = start.elapsed().as_secs_f64();
                    let start = Instant::now();
                    let (logits, rows) = h.forward(&windows)?;
                    std::hint::black_box(h.walk(&logits, &rows, &windows));
                    let verify = start.elapsed().as_secs_f64();
                    let start = Instant::now();
                    let restored = h.ring(&plans, Some(1))?;
                    let restore = start.elapsed().as_secs_f64();
                    if iteration >= case.warmup {
                        ts.push(if plans.iter().any(|(_, p)| p.save.is_some()) { save } else { 0.0 });
                        tv.push(verify);
                        tr.push(if restored { restore } else { 0.0 });
                    }
                }
                let (ts, tv, tr) = (median(ts)?, median(tv)?, median(tr)?);
                m.ts.insert((k, n), ts);
                m.tv.insert((w, n), tv);
                m.tr.insert((k, n), tr);
                let lanes_saved = plans.iter().filter(|(_, p)| p.save.is_some()).count();
                flat.push(serde_json::json!({"head": head, "figure": "tv", "w": w, "n": n, "seconds": tv}));
                flat.push(serde_json::json!({"head": head, "figure": "ts", "k": k, "n": n,
                    "seconds": ts, "lanes_saved": lanes_saved}));
                flat.push(serde_json::json!({"head": head, "figure": "tr", "k": k, "n": n,
                    "seconds": tr, "lanes_restored": lanes_saved}));
            }
        }
    }
    std::env::remove_var("HF2Q_FUSE_LMHEAD");
    let (drafter, drafter_json) = drafter_bound(&mut h, case.warmup, case.samples)?;

    let mut decisions = serde_json::Map::new();
    let mut batched_n1_needed = serde_json::Map::new();
    for (head, m) in &mut measurements {
        m.drafter = drafter.clone();
        let decision = decide_g0(m, &tick_seconds, &[1, 2, 3])?;
        let k = decision.k_star;
        let needed = (1..=LANES).any(|n| {
            m.ts[&(k, n)] + m.tr[&(k, n)] > 0.03 * m.tv[&(k + 1, n)]
        });
        batched_n1_needed.insert((*head).into(), needed.into());
        decisions.insert((*head).into(), serde_json::to_value(&decision)?);
    }
    let pass_both = decisions
        .values()
        .all(|d| d["pass"].as_bool() == Some(true));
    let receipt = serde_json::json!({
        "schema": "hf2q.q4-gemma-spec-serve.m0.v1",
        "source_commit": std::env::var("GIT_COMMIT_SHA").ok(),
        "case_file": path.to_string_lossy(),
        "case_sha256": hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&raw)),
        "model": case.model,
        "context": case.context,
        "lanes": h.lanes.iter().zip(&lane_sources).map(|(l, source)| serde_json::json!({
            "slot": l.slot.0, "pos0": l.pos0, "window": l.tail, "source": source}))
            .collect::<Vec<_>>(),
        "mlx_native_version": env!("HF2Q_MLX_NATIVE_VERSION"),
        "restore_pipeline_identity": format!("{:?}", h.model.ctx.split().1
            .pipeline_identity("kv_cache_restore_ring_bytes")),
        "warmup": case.warmup,
        "samples": case.samples,
        "class_a": class_a,
        "figures": flat,
        "drafter_bound": drafter_json,
        "tick_seconds_by_width": case.tick_seconds_by_width,
        "decision": decisions,
        "batched_n1_needed": batched_n1_needed,
        "ring_dispatches_per_lane": h.preimages[0].dispatches_per_lane(),
        "wrap_coverage": h.lanes.iter().zip(&wrapped).map(|(l, &w)| serde_json::json!({
            "slot": l.slot.0, "pos0": l.pos0, "wraps_ring": w,
            "ring_capacity": h.ring_capacity})).collect::<Vec<_>>(),
        "known_bias": "tv(w, n) at w ≥ 3 (K ≥ 2) is biased high: with more than 8 rows the dense \
            projections and the lm_head run one m=1 pass per row (design §5.3's m ≤ 8 grouping is \
            not implemented), so the provisional K* leans toward 1.",
        "g0_pass_both_head_paths": pass_both,
    });
    let out = case.output.join("m0-round-cost.json");
    std::fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt["decision"])?);
    println!("M0 receipt: {}", out.display());
    Ok(())
}

#[test]
#[ignore = "root supervised real Gemma 4 target; GPU lane and M0 case file required"]
fn gemma_spec_round_cost_curve() {
    run().unwrap();
}
