//! HF2Q_SPEC_DFLASH=1 CLI integration (ADR-030 iter-66 production wire-up).
//!
//! Provides a single entry-point [`try_dispatch_dflash_spec_decode`] that
//! cmd_generate calls right after prompt tokenization.  When the env flag
//! is unset, returns `Ok(None)` so the caller falls through to the
//! existing per-token decode loop.  When set, the helper loads the
//! z-lab DFlash drafter, runs
//! [`crate::inference::spec_decode::dflash::orchestrator::dispatch_dflash_generate`],
//! prints the generated text, and returns `Ok(Some(()))` so the caller
//! returns from cmd_generate immediately.
//!
//! ## Env vars
//!
//! - `HF2Q_SPEC_DFLASH` (= "1") — opt in to spec-decode for this run.
//! - `HF2Q_DFLASH_DRAFTER_PATH` — override the drafter directory.  When
//!   unset, the helper defaults to the HuggingFace cache path
//!   `~/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/<sha>`.
//! - `HF2Q_DFLASH_BLOCK_SIZE` — override the block size K+1 (default 8,
//!   Phase 1.5 measured optimum on M5 Max).
//!
//! ## Correctness vs performance
//!
//! Coherence (byte-identity vs single-token decode at temp=0) is GREEN
//! per `e2e_dispatch_dflash_generate_gemma4_26b` on real models.
//!
//! Performance is currently SLOWER than baseline single-token decode
//! (~10× slowdown at N=8 tokens on M5 Max) because the orchestrator's
//! Option C path re-prefills the full prefix from `start_pos=0` each
//! round.  Mission perf gate (≥1.07× hf2q baseline) requires Option A
//! (cross-length SDPA in `flash_attn_prefill`) — deferred to a future
//! iter.  The env flag is OPT-IN to preserve production perf parity
//! while enabling correctness validation on user workloads.

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Hugging Face cache path for the z-lab DFlash drafter snapshot.
/// Matches the snapshot pinned in
/// `inference::spec_decode::dflash::orchestrator::tests` (iter-63).
const DEFAULT_DRAFTER_SNAPSHOT: &str =
    "models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057";

/// Resolve the drafter directory.  Priority:
/// 1. `HF2Q_DFLASH_DRAFTER_PATH` env var (absolute path expected)
/// 2. `~/.cache/huggingface/hub/<DEFAULT_DRAFTER_SNAPSHOT>`
fn resolve_drafter_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("HF2Q_DFLASH_DRAFTER_PATH") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("HOME").context("HOME env var not set")?;
    Ok(PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/{DEFAULT_DRAFTER_SNAPSHOT}"
    )))
}

/// Resolve `block_size` (K+1).  Phase 1.5 default = 8 (K=7).
fn resolve_block_size() -> Result<u32> {
    match std::env::var("HF2Q_DFLASH_BLOCK_SIZE") {
        Err(_) => Ok(8),
        Ok(s) => {
            let n: u32 = s
                .parse()
                .with_context(|| format!("HF2Q_DFLASH_BLOCK_SIZE must be integer; got {s:?}"))?;
            anyhow::ensure!(n >= 2, "HF2Q_DFLASH_BLOCK_SIZE must be ≥ 2; got {n}");
            Ok(n)
        }
    }
}

/// Dispatch DFlash spec-decode when `HF2Q_SPEC_DFLASH=1` is set.
///
/// - Returns `Ok(None)` when the env flag is unset; caller continues
///   with the standard prefill + per-token decode path.
/// - Returns `Ok(Some(()))` when spec-decode ran to completion; the
///   helper has printed the decoded text to stdout already.
/// - Returns `Err(_)` if the drafter cannot be loaded or
///   `dispatch_dflash_generate` fails.
pub fn try_dispatch_dflash_spec_decode(
    target: &mut crate::serve::forward_mlx::MlxModelWeights,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    ignore_eos: bool,
    tokenizer: &tokenizers::Tokenizer,
    gpu: &mut crate::serve::gpu::GpuContext,
) -> Result<Option<()>> {
    if std::env::var("HF2Q_SPEC_DFLASH").as_deref() != Ok("1") {
        return Ok(None);
    }
    eprintln!("[HF2Q_SPEC_DFLASH=1] loading DFlash drafter — coherent at temp=0, SLOWER than baseline (Option A pending for perf parity)");

    use crate::inference::spec_decode::dflash::{
        config::DFlashConfig,
        kv_cache::DFlashKvCache,
        orchestrator::dispatch_dflash_generate,
        tensors::DFlashModelTensors,
        weights::{DFlashWeights, DFlashWeightsFile},
    };

    let drafter_dir = resolve_drafter_path()?;
    if !drafter_dir.is_dir() {
        anyhow::bail!(
            "HF2Q_SPEC_DFLASH=1 but drafter dir {} does not exist. \
             Set HF2Q_DFLASH_DRAFTER_PATH or fetch \
             z-lab/gemma-4-26B-A4B-it-DFlash from HuggingFace first.",
            drafter_dir.display(),
        );
    }
    let cfg_path = drafter_dir.join("config.json");
    let weights_path = drafter_dir.join("model.safetensors");
    for p in [&cfg_path, &weights_path] {
        anyhow::ensure!(
            p.exists(),
            "DFlash drafter artifact missing: {}",
            p.display()
        );
    }

    // When --ignore-eos is set on the CLI, the per-token decode loop
    // bypasses the eos_token_ids check.  Mirror that by passing an
    // empty slice into dispatch_dflash_generate so it generates the
    // full max_new_tokens regardless of EOS emission.
    let effective_eos: &[u32] = if ignore_eos { &[] } else { eos_token_ids };

    let block_size = resolve_block_size()?;
    let t_load = std::time::Instant::now();
    let drafter_cfg = DFlashConfig::from_json_path(&cfg_path)
        .context("parse DFlash drafter config.json")?;
    let drafter_file =
        DFlashWeightsFile::open(&weights_path).context("open DFlash drafter safetensors")?;
    let drafter_weights = DFlashWeights::load(drafter_file.bytes(), &drafter_cfg)
        .context("validate + load DFlash drafter weights")?;
    let drafter_tensors = {
        let (exec, _reg) = gpu.split();
        DFlashModelTensors::upload(exec.device(), &drafter_cfg, &drafter_weights)
            .context("upload DFlash drafter weights to GPU")?
    };
    // Drafter cache capacity must cover the maximum prefix length the
    // drafter will see — bounded by the target's max_new_tokens plus
    // prompt length.  The orchestrator's Option C re-prefills the full
    // ctx each round, and `DFlashKvCache::reset()` is called between
    // rounds, so capacity only needs to fit ONE round's ctx (=
    // prompt_len + max_new_tokens at worst).  Use a generous bound so
    // long contexts don't fail mid-generation.
    let drafter_cache_cap = (prompt_tokens.len() + max_new_tokens + 32).max(2048) as u32;
    let mut drafter_cache = {
        let (exec, _reg) = gpu.split();
        DFlashKvCache::new(exec.device(), &drafter_cfg, drafter_cache_cap)
            .context("allocate DFlash drafter KV cache")?
    };
    eprintln!(
        "[HF2Q_SPEC_DFLASH] drafter loaded in {:.2}s (config={}, cache_cap={drafter_cache_cap})",
        t_load.elapsed().as_secs_f64(),
        cfg_path.display(),
    );

    let t_gen = std::time::Instant::now();
    let output_tokens = dispatch_dflash_generate(
        target,
        &drafter_tensors,
        &mut drafter_cache,
        &drafter_cfg,
        prompt_tokens,
        max_new_tokens,
        block_size,
        effective_eos,
        gpu,
    )
    .context("dispatch_dflash_generate")?;
    let gen_elapsed = t_gen.elapsed();

    let new_tokens = &output_tokens[prompt_tokens.len()..];
    let decoded = tokenizer
        .decode(new_tokens, /*skip_special=*/ false)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"));
    println!("{decoded}");
    eprintln!(
        "[HF2Q_SPEC_DFLASH] {} new tokens in {:.2}s ({:.1} tok/s)",
        new_tokens.len(),
        gen_elapsed.as_secs_f64(),
        new_tokens.len() as f64 / gen_elapsed.as_secs_f64().max(1e-6),
    );

    Ok(Some(()))
}

/// ADR-034 task #78 Step 4 (2026-05-21) — default Qwen35 DFlash drafter
/// path. The z-lab drafter ships at this absolute location after
/// `hf-download` (see /opt/hf2q/models/dflash-drafters/README).
///
/// Default chosen as the Qwen 3.6 27B dense drafter — matches the
/// production Cell A 27B target. Users can override via
/// `HF2Q_DFLASH_DRAFTER_PATH` for the 35B-A3B variant or any other
/// future drafter.
const DEFAULT_QWEN35_DRAFTER_DIR: &str =
    "/opt/hf2q/models/dflash-drafters/z-lab__Qwen3.6-27B-DFlash";

/// Resolve the Qwen35 DFlash drafter directory.
fn resolve_qwen35_drafter_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("HF2Q_DFLASH_DRAFTER_PATH") {
        return Ok(PathBuf::from(p));
    }
    Ok(PathBuf::from(DEFAULT_QWEN35_DRAFTER_DIR))
}

/// ADR-034 task #78 Step 4 (2026-05-21) — dispatch Qwen35 DFlash
/// spec-decode when `HF2Q_SPEC_DFLASH=1` is set, from the Qwen35-side
/// `cmd_generate_qwen35` call path.
///
/// Mirrors [`try_dispatch_dflash_spec_decode`] (which handles the
/// MlxModelWeights / Gemma 4 target), adapted for Qwen35Model +
/// HybridKvCache via the [`Qwen35DFlashTarget`] wrapper.
///
/// Returns `Ok(None)` when `HF2Q_SPEC_DFLASH` is unset (caller falls
/// through to the standard SpecDecode / decode path). Returns
/// `Ok(Some(()))` after writing the decoded text to stdout. Returns
/// `Err(_)` for drafter load failures or
/// `dispatch_qwen35_dflash_generate` failures.
///
/// ## Env vars
///
/// - `HF2Q_SPEC_DFLASH` (= "1") — opt in.
/// - `HF2Q_DFLASH_DRAFTER_PATH` — override the drafter directory.
///   Default = `/opt/hf2q/models/dflash-drafters/z-lab__Qwen3.6-27B-DFlash`.
/// - `HF2Q_DFLASH_BLOCK_SIZE` — override the block size K+1 (default
///   reads `drafter_cfg.block_size`, falling back to 8 if absent).
///
/// ## Correctness vs performance
///
/// At temperature=0 (greedy), output is byte-identical to single-token
/// Qwen35 decode per the orchestrator's `step_round_from_argmaxes`
/// + greedy-byte-identity proof (see `qwen35_orchestrator.rs` module
/// doc).
///
/// Performance is UNVALIDATED at the time of wiring (Step 4). Cell C
/// Gemma 4 DFlash was empirically 4.8x SLOWER than baseline because the
/// MlxModelWeights orchestrator uses Option C re-prefill. The Qwen35
/// orchestrator uses Option A (xlen verify) which avoids the
/// re-prefill — first empirical bench result will determine whether
/// Qwen35 DFlash beats the current production winner (MTP K=1 BATCHED
/// MH at 25.6 tok/s on Qwen 3.6 27B).
pub fn try_dispatch_qwen35_dflash_spec_decode(
    model: &mut crate::inference::models::qwen35::model::Qwen35Model,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    ignore_eos: bool,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Option<()>> {
    if std::env::var("HF2Q_SPEC_DFLASH").as_deref() != Ok("1") {
        return Ok(None);
    }
    // ADR-034 task #78 Step 5 (2026-05-21) — empirical bench result at
    // HEAD e10946cf on Qwen 3.6 27B Q8_0 + 128 tokens + temp=0:
    //   BS=2:  16.1 tok/s    (best DFlash result)
    //   BS=4:  12.3 tok/s
    //   BS=8:   8.7 tok/s
    //   BS=16:  3.8 tok/s    (drafter default)
    //   ----
    //   Base MTP K=1 BATCHED MH: 30.2 tok/s @ 92.4% accept
    //
    // DFlash on Qwen35 is currently 0.53x the production winner at the
    // optimal block_size. Root cause: K+1 batched verify uses the BF16
    // prefill_resume kernel which is slower per-token than the F32
    // flash_attn_vec single-token decode used by MTP K=1. Task #89
    // (forward_gpu_batched_decode for seq_len[2,8]) is the prerequisite
    // to close this gap.
    //
    // Correctness: verified greedy-coherent (orchestrator emitted a
    // valid haiku at BS=16 vs base path's degenerate "test coverage:"
    // loop on the same prompt — see commit e10946cf bench data).
    //
    // The flag is kept opt-in (default OFF) so future improvements can
    // be validated without changing production behavior.
    eprintln!(
        "[HF2Q_SPEC_DFLASH=1 qwen35] WARNING: empirical bench (2026-05-21) shows \
         DFlash at best ~0.53x production MTP K=1 BATCHED MH path (16 vs 30 tok/s \
         on Qwen 3.6 27B Q8_0). Root cause: BF16 prefill_resume per-token cost; \
         task #89 (batched_decode for seq_len[2,8]) is the prerequisite to close \
         the gap. Use HF2Q_SPEC_DECODE=1 + --temperature 0.5 for the production \
         winner instead."
    );

    use crate::inference::spec_decode::dflash::{
        config::DFlashConfig,
        kv_cache::DFlashKvCache,
        qwen35_orchestrator::dispatch_qwen35_dflash_generate,
        qwen35_target::Qwen35DFlashTarget,
        tensors::DFlashModelTensors,
        weights::{DFlashWeights, DFlashWeightsFile},
    };

    let drafter_dir = resolve_qwen35_drafter_path()?;
    if !drafter_dir.is_dir() {
        anyhow::bail!(
            "HF2Q_SPEC_DFLASH=1 (qwen35 path) but drafter dir {} does not exist. \
             Set HF2Q_DFLASH_DRAFTER_PATH or fetch \
             z-lab/Qwen3.6-27B-DFlash from HuggingFace first.",
            drafter_dir.display(),
        );
    }
    let cfg_path = drafter_dir.join("config.json");
    let weights_path = drafter_dir.join("model.safetensors");
    for p in [&cfg_path, &weights_path] {
        anyhow::ensure!(
            p.exists(),
            "DFlash drafter artifact missing: {}",
            p.display()
        );
    }

    // --ignore-eos parity with Gemma path: pass empty eos slice so the
    // orchestrator generates the full max_new_tokens regardless of EOS.
    let effective_eos: &[u32] = if ignore_eos { &[] } else { eos_token_ids };

    let t_load = std::time::Instant::now();
    let drafter_cfg = DFlashConfig::from_json_path(&cfg_path)
        .context("parse Qwen35 DFlash drafter config.json")?;

    // block_size resolution: prefer drafter_cfg.block_size (set by
    // z-lab — Qwen 3.6 27B drafter ships block_size=16), allow env
    // override.
    let block_size: u32 = match std::env::var("HF2Q_DFLASH_BLOCK_SIZE") {
        Ok(s) => {
            let n: u32 = s.parse().with_context(|| {
                format!("HF2Q_DFLASH_BLOCK_SIZE must be integer; got {s:?}")
            })?;
            anyhow::ensure!(n >= 2, "HF2Q_DFLASH_BLOCK_SIZE must be >= 2; got {n}");
            n
        }
        Err(_) => {
            // drafter_cfg.block_size is usize; the orchestrator + GPU
            // dispatch all take u32. Validate it fits.
            let bs = drafter_cfg.block_size;
            anyhow::ensure!(
                bs >= 2 && bs <= u32::MAX as usize,
                "drafter_cfg.block_size out of range: {bs}"
            );
            bs as u32
        }
    };

    let drafter_file = DFlashWeightsFile::open(&weights_path)
        .context("open Qwen35 DFlash drafter safetensors")?;
    let drafter_weights = DFlashWeights::load(drafter_file.bytes(), &drafter_cfg)
        .context("validate + load Qwen35 DFlash drafter weights")?;
    // Drafter weights upload to the SAME MlxDevice as the Qwen35Model
    // (avoids the "MlxBufferPool cannot mix residency-enabled devices"
    // failure). The Qwen35Model holds its GPU state in a thread-local
    // cache, so we go through `with_gpu_cache_mut` to fetch the device.
    model
        .ensure_gpu_cache_primed()
        .context("ensure_gpu_cache_primed before drafter upload")?;
    let drafter_tensors = model.with_gpu_cache_mut(|device, _reg| {
        DFlashModelTensors::upload(device, &drafter_cfg, &drafter_weights)
            .context("upload Qwen35 DFlash drafter weights to GPU")
    })?;
    // Drafter cache capacity: worst case is prompt_len + max_new_tokens
    // committed tokens fed as drafter context. Use the +32 buffer +
    // 2048 floor as the Gemma path does.
    //
    // Codex /cfa 2026-05-21 High: bound against model max_position_embeddings
    // and use checked_add to avoid usize overflow or u32 truncation at
    // huge --max-tokens.
    let max_pos = model.cfg.max_position_embeddings as usize;
    let bounded_max_new = max_new_tokens.min(max_pos);
    let drafter_cache_cap_usize = prompt_tokens
        .len()
        .checked_add(bounded_max_new)
        .and_then(|s| s.checked_add(32))
        .ok_or_else(|| anyhow::anyhow!("drafter_cache_cap overflow"))?
        .max(2048);
    anyhow::ensure!(
        drafter_cache_cap_usize <= u32::MAX as usize,
        "drafter_cache_cap {} > u32::MAX",
        drafter_cache_cap_usize,
    );
    let drafter_cache_cap = drafter_cache_cap_usize as u32;
    let mut drafter_cache = model.with_gpu_cache_mut(|device, _reg| {
        DFlashKvCache::new(device, &drafter_cfg, drafter_cache_cap)
            .context("allocate Qwen35 DFlash drafter KV cache")
    })?;
    eprintln!(
        "[HF2Q_SPEC_DFLASH qwen35] drafter loaded in {:.2}s (cfg={}, block_size={block_size}, \
         cache_cap={drafter_cache_cap}, target_layers={:?})",
        t_load.elapsed().as_secs_f64(),
        cfg_path.display(),
        drafter_cfg.target_layer_ids,
    );

    // Allocate a fresh HybridKvCache sized for the worst-case forward.
    // The orchestrator's per-round verify writes K+1 (=block_size)
    // positions starting at output.len()-1. Worst case at the FINAL
    // round: output.len() = prompt_len + max_new_tokens, so the cache
    // needs `prompt_len + max_new_tokens + block_size` slots.
    //
    // Codex /cfa 2026-05-21 High: clamp against model
    // max_position_embeddings + use checked_add to avoid usize overflow
    // and u32 truncation. We share `bounded_max_new` with the drafter
    // cache calc above so both bounds agree.
    let kv_max_seq_usize = prompt_tokens
        .len()
        .checked_add(bounded_max_new)
        .and_then(|s| s.checked_add(block_size as usize))
        .ok_or_else(|| anyhow::anyhow!("kv_max_seq overflow"))?;
    anyhow::ensure!(
        kv_max_seq_usize <= u32::MAX as usize,
        "kv_max_seq {} > u32::MAX",
        kv_max_seq_usize,
    );
    anyhow::ensure!(
        kv_max_seq_usize <= max_pos + block_size as usize,
        "kv_max_seq {} exceeds model max_position_embeddings+block_size = {}",
        kv_max_seq_usize,
        max_pos + block_size as usize,
    );
    let kv_max_seq = kv_max_seq_usize as u32;
    let mut kv_cache = model.with_gpu_cache_mut(|device, _reg| {
        crate::inference::models::qwen35::kv_cache::HybridKvCache::new(
            &model.cfg,
            device,
            kv_max_seq,
            1,
        )
        .context("alloc Qwen35 DFlash HybridKvCache")
    })?;

    // The orchestrator's `gpu` argument is required by the trait
    // signature (`DFlashTarget::forward_decode_verify_batched`) but
    // is IGNORED by the Qwen35 impl (Qwen35Model uses its own
    // thread-local GPU_CACHE — see Qwen35DFlashTarget::
    // forward_decode_verify_batched signature `_gpu`). We construct a
    // fresh GpuContext just to satisfy the type-checker. The
    // residency-enabled second MlxDevice coexists with the model's
    // GPU_CACHE device because each device owns its own
    // ResidencySet (see /opt/mlx-native/src/device.rs:46 — no shared
    // residency state across MlxDevice instances). Existing Qwen35
    // diagnostic paths (`qwen35_export_layer_states`,
    // `qwen35_dump_layer_states` in serve/mod.rs) already use this
    // pattern.
    let mut gpu = crate::serve::gpu::GpuContext::new()
        .map_err(|e| anyhow::anyhow!("GpuContext::new for trait API: {e}"))?;

    let t_gen = std::time::Instant::now();
    let mut target = Qwen35DFlashTarget::new(model, &mut kv_cache);
    let output_tokens = dispatch_qwen35_dflash_generate(
        &mut target,
        &drafter_tensors,
        &mut drafter_cache,
        &drafter_cfg,
        prompt_tokens,
        max_new_tokens,
        block_size,
        effective_eos,
        &mut gpu,
    )
    .context("dispatch_qwen35_dflash_generate")?;
    let gen_elapsed = t_gen.elapsed();

    let new_tokens = &output_tokens[prompt_tokens.len()..];
    let decoded = tokenizer
        .decode(new_tokens, /*skip_special=*/ false)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"));
    println!("{decoded}");
    eprintln!(
        "[HF2Q_SPEC_DFLASH qwen35] {} new tokens in {:.2}s ({:.1} tok/s)",
        new_tokens.len(),
        gen_elapsed.as_secs_f64(),
        new_tokens.len() as f64 / gen_elapsed.as_secs_f64().max(1e-6),
    );

    Ok(Some(()))
}

/// Resolve `K` for the n-gram proposer.  Default = 3 per
/// ADR-029 Phase 1 / vLLM literature (`k=3, max_ngram=3, min_ngram=1`).
fn resolve_ngram_k() -> Result<u32> {
    match std::env::var("HF2Q_SPEC_NGRAM_K") {
        Err(_) => Ok(3),
        Ok(s) => {
            let n: u32 = s
                .parse()
                .with_context(|| format!("HF2Q_SPEC_NGRAM_K must be integer; got {s:?}"))?;
            anyhow::ensure!(n >= 1, "HF2Q_SPEC_NGRAM_K must be ≥ 1; got {n}");
            Ok(n)
        }
    }
}

fn resolve_ngram_min() -> Result<u32> {
    match std::env::var("HF2Q_SPEC_NGRAM_MIN") {
        Err(_) => Ok(1),
        Ok(s) => Ok(s.parse().with_context(|| {
            format!("HF2Q_SPEC_NGRAM_MIN must be integer; got {s:?}")
        })?),
    }
}

fn resolve_ngram_max() -> Result<u32> {
    match std::env::var("HF2Q_SPEC_NGRAM_MAX") {
        Err(_) => Ok(3),
        Ok(s) => Ok(s.parse().with_context(|| {
            format!("HF2Q_SPEC_NGRAM_MAX must be integer; got {s:?}")
        })?),
    }
}

/// Dispatch n-gram spec-decode when `HF2Q_SPEC_NGRAM=1` is set
/// (ADR-030 iter-216 Plan B — pure-CPU proposer alternative to the
/// DFlash drafter, which had 0% acceptance against the ara-abliterated
/// APEX-Q5_K_M target per iter-212 measurement).
///
/// - Returns `Ok(None)` when the env flag is unset; caller continues
///   with the standard prefill + per-token decode path (or DFlash if
///   THAT flag is set instead).
/// - Returns `Ok(Some(()))` when ngram spec-decode ran to completion.
/// - Returns `Err(_)` if `dispatch_ngram_generate` fails.
///
/// Env knobs:
/// - `HF2Q_SPEC_NGRAM` (= "1") — opt in.
/// - `HF2Q_SPEC_NGRAM_K` (default 3) — draft length per round.
/// - `HF2Q_SPEC_NGRAM_MIN` (default 1) — min n-gram size to match.
/// - `HF2Q_SPEC_NGRAM_MAX` (default 3) — max n-gram size to match.
/// - `HF2Q_SPEC_NGRAM_PROFILE` (= "1") — print per-round timing on exit.
/// - `HF2Q_DFLASH_XLEN_SDPA` (= "1") — engage Option A cross-length
///   SDPA verify path (requires `HF2Q_FULL_F16_KV=1`).  Shared flag
///   with DFlash since both orchestrators use the same target verify
///   pipeline.
pub fn try_dispatch_ngram_spec_decode(
    target: &mut crate::serve::forward_mlx::MlxModelWeights,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    ignore_eos: bool,
    tokenizer: &tokenizers::Tokenizer,
    gpu: &mut crate::serve::gpu::GpuContext,
) -> Result<Option<()>> {
    if std::env::var("HF2Q_SPEC_NGRAM").as_deref() != Ok("1") {
        return Ok(None);
    }

    let k = resolve_ngram_k()?;
    let min_ngram = resolve_ngram_min()?;
    let max_ngram = resolve_ngram_max()?;
    eprintln!(
        "[HF2Q_SPEC_NGRAM=1] enabled — pure-CPU ngram proposer K={k} \
         min_ngram={min_ngram} max_ngram={max_ngram} \
         (workload-specific; ~80% accept needed to beat baseline)"
    );

    let effective_eos: &[u32] = if ignore_eos { &[] } else { eos_token_ids };

    let t_gen = std::time::Instant::now();
    let output_tokens = crate::inference::spec_decode::ngram_orchestrator::dispatch_ngram_generate(
        target,
        prompt_tokens,
        max_new_tokens,
        k,
        min_ngram,
        max_ngram,
        effective_eos,
        gpu,
    )
    .context("dispatch_ngram_generate")?;
    let gen_elapsed = t_gen.elapsed();

    let new_tokens = &output_tokens[prompt_tokens.len()..];
    let decoded = tokenizer
        .decode(new_tokens, /*skip_special=*/ false)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"));
    println!("{decoded}");
    eprintln!(
        "[HF2Q_SPEC_NGRAM] {} new tokens in {:.2}s ({:.1} tok/s)",
        new_tokens.len(),
        gen_elapsed.as_secs_f64(),
        new_tokens.len() as f64 / gen_elapsed.as_secs_f64().max(1e-6),
    );

    Ok(Some(()))
}
