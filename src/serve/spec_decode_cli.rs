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
    let xlen_sdpa = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
    // Mirror forward_mlx.rs:1322-1325's accepted truthy values for
    // HF2Q_FULL_F16_KV so this guard doesn't reject a user-configured
    // `HF2Q_FULL_F16_KV=true` that the V-alloc path WOULD accept.
    let full_f16_kv = std::env::var("HF2Q_FULL_F16_KV")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    // ADR-034 Cell C — fail-loud early instead of crashing mid-layer-loop with
    // "xlen SDPA L0: V is not F16 (got U8); HF2Q_FULL_F16_KV=1 required".
    // The xlen SDPA path at forward_prefill_batched.rs:1475-1501 requires F16
    // V cache and is incompatible with TQ-HB 8-bit V quantization. Catch the
    // user error here BEFORE we load the drafter (which takes ~80ms) instead
    // of letting it crash after the first round's verify forward starts.
    if xlen_sdpa && !full_f16_kv {
        anyhow::bail!(
            "HF2Q_DFLASH_XLEN_SDPA=1 requires HF2Q_FULL_F16_KV=1 (xlen cross-length SDPA \
             path needs F16 V cache; default TQ-HB 8-bit V quantization is incompatible). \
             Set both env vars, or unset HF2Q_DFLASH_XLEN_SDPA to fall back to Option C \
             (re-prefill from start_pos=0 each round, slower but doesn't require F16 V).",
        );
    }
    if xlen_sdpa {
        // ADR-034 2026-05-22: Option A xlen SDPA path is WIRED + delivers
        // 1.62× over Option C on Gemma 4 26B-A4B-it Q5_K_M code-gen (3-rep
        // paired bench at HEAD 7da12c37: Option A 45.7 t/s vs Option C
        // 28.2 t/s) AND produces output byte-identical to base autoregressive
        // for ~135 tokens then diverges via single-token argmax flip at
        // temp=0 greedy (empirical at HEAD 6a8a8f6f 2026-05-22:
        // base "element" vs Option A "term" at gen token ~135, cascading).
        // Still 0.40× of base generation — drafter
        // overhead exceeds Gemma's compact autoregressive cost; production
        // parity needs drafter training or tree decoding.
        eprintln!(
            "[HF2Q_SPEC_DFLASH=1 + HF2Q_DFLASH_XLEN_SDPA=1] loading DFlash drafter — Option A \
             (cross-length SDPA), byte-identical to base for ~135 tokens then diverges via argmax flip at temp=0 greedy, 1.62× over Option C \
             on Gemma but still 0.40× of base generation (research-quality)"
        );
    } else {
        // Option C — historical default; slower AND text diverges from base
        // due to F16/BF16 accumulation order in re-prefill from start_pos=0.
        eprintln!(
            "[HF2Q_SPEC_DFLASH=1] loading DFlash drafter — Option C re-prefill, slower than \
             baseline + diverges from base autoregressive at temp=0; set HF2Q_DFLASH_XLEN_SDPA=1 \
             + HF2Q_FULL_F16_KV=1 for Option A (1.62× faster + byte-identical to base for ~135 tokens then diverges)"
        );
    }

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
/// At temperature=0 (greedy), the orchestrator's accept-walk emits
/// tokens consistent with its OWN batched verifier's argmax at each
/// accepted position. This is NOT byte-identical to single-token
/// Qwen35 decode output — empirical at HEAD 334008e9 shows divergence
/// at 32 tokens already (root cause: BF16 prefill_resume verifier vs
/// F32 flash_attn_vec single-token decode). See `qwen35_orchestrator.rs`
/// module doc for the full empirical evidence and ADR-034's "Output
/// divergence note" for the same phenomenon affecting MTP K=1 BATCHED.
///
/// Performance VALIDATED (post-task #95 closure 2026-05-22 at HEAD
/// 334008e9, 128 tok 3-rep paired): Qwen35 DFlash is research-quality —
/// 27B DFlash 23.17 t/s = 0.77x of MTP K=1 greedy production winner
/// (was 0.62x pre-task-#95; the compile-drafter lever shipped +26%
/// cumulative); 35B-A3B DFlash 42.7 t/s = 0.31x of base autoregressive
/// (136.1) or 0.43x of MTP K=1 BATCHED forced (98.6). Both show
/// duplication artifacts (row-N kernel divergence). The Qwen35
/// orchestrator uses Option A (xlen verify) so the per-round forward
/// is incremental, NOT the Option C re-prefill that makes Gemma Cell C
/// 0.25x of base (per ADR-034 line 38 empirical at HEAD 7da12c37).
/// Closure to production parity needs drafter training
/// or tree decoding (multi-week).
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
    // ADR-034 task #95 closure + 35B-A3B empirical (2026-05-22) — bench
    // result at HEAD 334008e9 on Qwen 3.6 27B Q8_0 + Qwen 3.5 35B-A3B Q4_K_M,
    // 128 tokens, temp=0 greedy, 3-rep paired:
    //
    //   Qwen 3.6 27B dense MTP target + 27B DFlash drafter:
    //     DFlash BS=4: 23.17 tok/s  (post-task #95 +26% over pre-#95 18.4)
    //     MTP K=1 greedy code-gen winner: 30.0 tok/s @ 91% accept
    //     DFlash / MTP greedy = 0.77x  (was 0.62x pre-#95)
    //
    //   Qwen 3.5 35B-A3B MoE MTP target + 35B-A3B DFlash drafter:
    //     DFlash BS=4: 42.7 tok/s
    //     Base autoregressive: 136.1 tok/s
    //     DFlash / base = 0.31x
    //     MTP K=1 BATCHED forced (MoE): 98.6 tok/s
    //     DFlash / MTP K=1 = 0.43x
    //
    // Coherence: both targets produce coherence-degraded text at temp=0
    // greedy. 27B Fibonacci 128-tok run becomes garbled at ~30 tok
    // (`a, b =  a, b =  a, b = b, a + b = b, a + b, a + b` + return wrong
    // type), hits EOS, loops back into `<|im_end|><|im_start|>user` chat
    // markers, and duplicates lines in the thinking process. 35B-A3B
    // Fibonacci 128-tok run shows duplicated `def fibonacci(n):` +
    // duplicated `return [0]` lines but stays closer to the structure.
    // Same row-N kernel divergence pattern as K=N MTP (see ADR-034
    // §K=N CORRECTION block). NOT a DN-rollback issue (task #90 SHIPPED).
    //
    // Closure to production parity needs drafter training (smaller/faster
    // drafter) or tree decoding (EAGLE-2/Medusa) bypassing the chained
    // accept-walk. Both multi-week scope.
    //
    // The flag is kept opt-in (default OFF) so users default to the
    // production winner (MTP K=1 greedy code-gen / MH temp=0.5 essay).
    eprintln!(
        "[HF2Q_SPEC_DFLASH=1 qwen35] WARNING: empirical bench (HEAD 334008e9 2026-05-22, \
         128 tok 3-rep paired) — DFlash is research-quality on Qwen35: 27B DFlash \
         23.17 t/s = 0.77x of MTP K=1 greedy (was 0.62x pre-task #95); 35B-A3B DFlash \
         42.7 t/s = 0.31x of base (136.1) or 0.43x of MTP K=1 BATCHED (98.6). Output \
         is coherence-degraded at longer lengths (27B Fibonacci 128-tok becomes \
         garbled `a, b = a, b = ...` + hits EOS + loops into chat markers; 35B-A3B \
         shows duplicated lines). Row-N kernel divergence vs single-token decode. \
         For production use HF2Q_SPEC_DECODE=1 --temperature 0 (code-gen 1.37x base) \
         or 0.5 (essay 1.26x base)."
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
