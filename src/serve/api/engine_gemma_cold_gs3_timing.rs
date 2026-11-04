//! ADR-063 Q4-GEMMA-COLD G-S3 (design §7.1): no-boundary cold batch
//! transaction timing.
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF; GPU
//! lane and quiet host required. `scripts/q4_gemma_cold_gs3.sh` builds this
//! harness on the candidate (per-sequence cold attention) and, unchanged, on
//! the base commit (block-diagonal route): it only calls
//! `forward_prefill_batched_multi_seq`, which both binaries provide, so each
//! build times its own cold multi-sequence route. Every record is one
//! transaction, printed as `GEMMA_COLD_GS3_RECORD {json}`.
//!
//! Sets: four 465-row prompts, and eight 64-row prompts; `HF2Q_GEMMA_COLD_GS3_REPEATS`
//! timed repeats each (default 10) after one untimed warm-up.
use super::*;

fn gs3_prompt(lane: usize, len: usize, vocab: usize) -> Vec<u32> {
    let span = vocab - 2_048;
    (0..len)
        .map(|i| (1_024 + (i * 7_919 + lane * 104_729 + 17) % span) as u32)
        .collect()
}

#[test]
#[ignore = "official Gemma 4 GGUF: G-S3 cold multi-sequence transaction timing (GPU lane)"]
fn gemma_cold_gs3_multi_seq_timing() {
    let path = PathBuf::from(
        std::env::var("HF2Q_GEMMA4_GGUF").expect("HF2Q_GEMMA4_GGUF names the official GGUF"),
    );
    let repeats: usize = std::env::var("HF2Q_GEMMA_COLD_GS3_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let mut loaded = LoadedModel::load(&LoadOptions {
        model_path: path,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
    })
    .expect("load official Gemma GGUF");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    g.provision_multi_seq_kv_for_slot_aware(8)
        .expect("provision 8 slots");
    let scaffold = g.multi_seq_kv_hybrid.take().expect("hybrid scaffold");
    let vocab = g.weights.vocab_size;
    for (set, lanes, rows) in [("4x465", 4usize, 465usize), ("8x64", 8, 64)] {
        let seqs: Vec<(Vec<u32>, SlotId)> = (0..lanes)
            .map(|lane| (gs3_prompt(lane, rows, vocab), SlotId(lane as u32)))
            .collect();
        for repeat in 0..=repeats {
            let started = Instant::now();
            let output = g
                .weights
                .forward_prefill_batched_multi_seq(&seqs, &scaffold, 16, &mut g.ctx)
                .expect("cold multi-sequence prefill");
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(output.first_tokens.len(), lanes);
            if repeat > 0 {
                println!(
                    "GEMMA_COLD_GS3_RECORD {{\"set\":\"{set}\",\"lanes\":{lanes},\"rows\":{rows},\"repeat\":{repeat},\"ms\":{ms:.4},\"first_tokens\":{:?}}}",
                    output.first_tokens
                );
            }
        }
    }
    g.multi_seq_kv_hybrid = Some(scaffold);
}
