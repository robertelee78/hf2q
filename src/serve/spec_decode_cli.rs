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
