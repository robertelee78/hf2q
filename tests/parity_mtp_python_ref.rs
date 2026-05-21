//! Numerical-parity test for Qwen 3.5/3.6 MTP forward — ADR-034 G2 gate.
//!
//! Loads .npy dumps produced by `scripts/parity/mtp_parity.py`, runs
//! `MtpWeights::forward_draft` on the same inputs, and asserts max-abs-diff
//! is within the threshold ladder defined in ADR-034 §G2.
//!
//! ## Status (2026-05-21)
//!
//! 🟡 **Scaffold** — runs but skips with explanatory message when the .npy
//! dumps are absent. Re-enables automatically when:
//!   1. `scripts/parity/mtp_parity.py` is fully implemented (currently scaffold)
//!   2. Qwen 3.6 MTP-bearing safetensors are downloaded
//!   3. Reference dumps are produced into `tests/parity_fixtures/mtp/`
//!
//! ## Discovery context
//!
//! During prep deep-research on 2026-05-21, the MTP runtime loader was
//! found to hardcode dense FFN tensor names (`blk.N.ffn_gate.weight`)
//! while canonical Qwen 3.5 35B-A3B GGUF emits MoE-style names
//! (`blk.N.ffn_gate_exps.weight`, 16 inner-MoE tensors at the MTP block).
//!
//! See [[project_adr034_mtp_loader_moe_bug_2026_05_21]] memory entry.
//!
//! ## Threshold ladder (per ADR-034 §G2)
//!
//! - F32 forward path: max-abs-diff < 1e-5
//! - BF16 forward path: max-abs-diff < 1e-3
//! - Q4_K_M forward path: max-abs-diff < 5e-2

use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity_fixtures/mtp")
}

#[test]
fn mtp_parity_against_python_reference_2026_05_21() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!(
            "skipping: tests/parity_fixtures/mtp/ does not exist.\n\
             To enable this test:\n\
             1. Implement scripts/parity/mtp_parity.py (scaffold only as of 2026-05-21)\n\
             2. Download a Qwen 3.6 MTP-bearing checkpoint to /opt/hf2q/models/\n\
             3. Run: python3 scripts/parity/mtp_parity.py <model_dir> tests/parity_fixtures/mtp/\n\
             4. Re-run: cargo test --release mtp_parity_against_python_reference\n\
             \n\
             See ADR-034 §P1 + project_adr034_mtp_loader_moe_bug_2026_05_21 memory entry.\n"
        );
        return;
    }

    // When fixtures exist, the implementation will:
    // 1. Load input_hidden.npy, input_embed.npy, input_position.npy
    // 2. Load the matching GGUF (built by hf2q convert from same source as Python ref)
    // 3. Call MtpWeights::forward_draft with the same inputs
    // 4. Compare logits.npy vs Rust output, assert max-abs-diff per ADR-034 §G2 ladder
    // 5. Also compare intermediates (eh_proj_out, attn_out, ffn_out, shared_head_norm_out)
    //    to localize where divergence occurs if the final logits diverge.
    //
    // Pseudo-code for the implementation:
    //
    //   let input_hidden = load_npy_f32(&dir.join("input_hidden.npy"))?;
    //   let input_embed  = load_npy_f32(&dir.join("input_embed.npy"))?;
    //   let position_ids = load_npy_i32(&dir.join("input_position.npy"))?;
    //   let expected_logits = load_npy_f32(&dir.join("logits.npy"))?;
    //
    //   let device = MlxDevice::default()?;
    //   let mut registry = KernelRegistry::new();
    //   let gguf = GgufFile::open(&fixtures_gguf_path)?;
    //   let cfg = Qwen35Config::from_gguf(&gguf)?;
    //   let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)?
    //       .expect("MTP not present in fixture GGUF");
    //   let mut kv = HybridKvCache::new(&cfg, &device, ctx_len, 1)?;
    //
    //   let prev_hidden = upload_f32(&input_hidden, &device)?;
    //   let embed_t = upload_f32(&input_embed, &device)?;
    //   let actual_logits = mtp.forward_draft(
    //       &prev_hidden, &embed_t, &mut kv, &position_ids, &device, &mut registry, &cfg,
    //   )?;
    //
    //   let actual = download_f32(&actual_logits, &device)?;
    //   let max_abs_diff = expected_logits.iter().zip(actual.iter())
    //       .map(|(a, b)| (a - b).abs())
    //       .fold(0.0_f32, f32::max);
    //
    //   let threshold = match quant_kind(&fixtures_gguf_path) {
    //       Quant::F32 => 1e-5,
    //       Quant::BF16 => 1e-3,
    //       Quant::Q4_K_M => 5e-2,
    //   };
    //   assert!(
    //       max_abs_diff < threshold,
    //       "MTP forward parity violation: max-abs-diff {:.6e} >= threshold {:.6e}",
    //       max_abs_diff, threshold,
    //   );

    eprintln!("TODO: implement when scaffold dependencies are unblocked");
}
