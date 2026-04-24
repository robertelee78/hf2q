//! Token-embedding lookup and LM-head output projection — the non-layer
//! parts of the Qwen3.5 forward pass.
//!
//! Together with the per-layer references in [`super::full_attn`],
//! [`super::delta_net`], and [`super::ffn`], these complete a full-stack
//! CPU reference forward pass:
//!
//! ```text
//! tokens → embed_tokens  → hidden[0]
//!                        → for each layer: hidden[l+1] = hidden[l] + layer(hidden[l])
//!                        → apply_output_head(hidden[L]) → logits
//! ```
//!
//! Embedding and output head are standalone because they're shape-agnostic
//! (just table lookup + final norm + matmul); no Qwen3.5-specific quirks.
//! They're verified here so the integrated forward pass (P11) has nothing
//! left to debug at this layer.

/// Look up per-token embeddings.
///
/// * `tokens`: input token IDs of length `seq_len`.
/// * `token_embd`: embedding table of shape `[vocab_size, hidden_size]`
///   (row-major; row `i` is the embedding for token ID `i`).
/// * `vocab_size` + `hidden_size` — shape metadata.
///
/// Returns `[seq_len, hidden_size]` row-major.
pub fn embed_tokens(
    tokens: &[u32],
    token_embd: &[f32],
    vocab_size: u32,
    hidden_size: u32,
) -> Vec<f32> {
    let h = hidden_size as usize;
    let vocab = vocab_size as usize;
    assert_eq!(
        token_embd.len(),
        vocab * h,
        "token_embd: {} != vocab({}) * hidden({})",
        token_embd.len(),
        vocab,
        h
    );

    let seq = tokens.len();
    let mut out = vec![0.0f32; seq * h];
    for (t, &tok) in tokens.iter().enumerate() {
        let tok_idx = tok as usize;
        assert!(
            tok_idx < vocab,
            "token id {} out of range (vocab_size = {})",
            tok, vocab
        );
        let src = &token_embd[tok_idx * h..(tok_idx + 1) * h];
        out[t * h..(t + 1) * h].copy_from_slice(src);
    }
    out
}

/// Apply the final output head: RMSNorm then LM-head projection.
///
/// Given the residual stream at the top of the stack, produces per-token
/// logits over the vocabulary.
///
/// * `hidden`: `[seq_len, hidden_size]` — residual stream after the last layer.
/// * `output_norm_w`: `[hidden_size]` — final RMSNorm weight.
/// * `output_weight`: `[hidden_size, vocab_size]` — LM head (GGUF row-major
///   `[out_dim, in_dim]` convention, so row `i` is the logit weights that
///   produce vocab position `i`).
/// * `rms_norm_eps`: RMSNorm epsilon.
///
/// Returns `[seq_len, vocab_size]` row-major logits.
pub fn apply_output_head(
    hidden: &[f32],
    output_norm_w: &[f32],
    output_weight: &[f32],
    hidden_size: u32,
    vocab_size: u32,
    rms_norm_eps: f32,
) -> Vec<f32> {
    let h = hidden_size as usize;
    let v = vocab_size as usize;
    let seq = hidden.len() / h;

    assert_eq!(hidden.len(), seq * h);
    assert_eq!(output_norm_w.len(), h);
    assert_eq!(output_weight.len(), v * h);

    // 1. Final RMSNorm per-token over hidden dim.
    let mut normed = vec![0.0f32; seq * h];
    for t in 0..seq {
        let row = &hidden[t * h..(t + 1) * h];
        let sum_sq: f32 = row.iter().map(|x| x * x).sum();
        let inv = ((sum_sq / (h as f32)) + rms_norm_eps).sqrt().recip();
        for j in 0..h {
            normed[t * h + j] = row[j] * inv * output_norm_w[j];
        }
    }

    // 2. LM head projection: logits[t, i] = sum_j output_weight[i, j] * normed[t, j].
    //    output_weight row i has length h.
    let mut logits = vec![0.0f32; seq * v];
    for t in 0..seq {
        for i in 0..v {
            let mut acc = 0.0f32;
            for j in 0..h {
                acc += output_weight[i * h + j] * normed[t * h + j];
            }
            logits[t * v + i] = acc;
        }
    }
    logits
}

/// Greedy argmax over the last token's logits.
///
/// Convenience helper for inference loops where `temperature == 0`.
pub fn greedy_argmax_last_token(logits: &[f32], vocab_size: u32) -> u32 {
    let v = vocab_size as usize;
    assert!(logits.len() >= v);
    let last = &logits[logits.len() - v..];
    let (max_idx, _) = last
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i as u32, v)
            } else {
                (best_i, best_v)
            }
        });
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedding lookup produces the expected row from the table.
    #[test]
    fn embed_tokens_basic() {
        let vocab = 5u32;
        let h = 4u32;
        // token_embd[i] = [i*4+0, i*4+1, i*4+2, i*4+3]
        let token_embd: Vec<f32> = (0..(vocab * h)).map(|x| x as f32).collect();

        let tokens = vec![0u32, 2, 4, 1];
        let out = embed_tokens(&tokens, &token_embd, vocab, h);

        assert_eq!(out.len(), 4 * 4);
        // Token 0 → row 0 → [0, 1, 2, 3].
        assert_eq!(&out[0..4], &[0.0, 1.0, 2.0, 3.0]);
        // Token 2 → row 2 → [8, 9, 10, 11].
        assert_eq!(&out[4..8], &[8.0, 9.0, 10.0, 11.0]);
        // Token 4 → row 4 → [16, 17, 18, 19].
        assert_eq!(&out[8..12], &[16.0, 17.0, 18.0, 19.0]);
        // Token 1 → row 1 → [4, 5, 6, 7].
        assert_eq!(&out[12..16], &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    #[should_panic(expected = "token id")]
    fn embed_tokens_panics_on_out_of_range() {
        let vocab = 3u32;
        let h = 2u32;
        let token_embd = vec![0.0f32; (vocab * h) as usize];
        let tokens = vec![5u32]; // out of range
        let _ = embed_tokens(&tokens, &token_embd, vocab, h);
    }

    #[test]
    fn embed_tokens_empty_input() {
        let token_embd = vec![0.0f32; 12];
        let tokens: Vec<u32> = vec![];
        let out = embed_tokens(&tokens, &token_embd, 3, 4);
        assert_eq!(out.len(), 0);
    }

    /// Output head with `output_weight = identity` and `output_norm_w = 1`
    /// (plus eps→0 via very small contribution) produces logits that
    /// are the normalized hidden states themselves.
    #[test]
    fn output_head_identity_weight_returns_normalized_hidden() {
        let h = 4u32;
        let v = 4u32;
        let seq = 2usize;

        // Hidden: two tokens, identity-like values.
        let hidden: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            2.0, 2.0, 2.0, 2.0, // token 1
        ];
        let output_norm_w = vec![1.0f32; 4];
        // Identity matrix output_weight: row i has 1 at position i.
        let mut output_weight = vec![0.0f32; 16];
        for i in 0..4 {
            output_weight[i * 4 + i] = 1.0;
        }

        let eps = 1e-12;
        let logits = apply_output_head(&hidden, &output_norm_w, &output_weight, h, v, eps);
        assert_eq!(logits.len(), seq * v as usize);

        // Expected token 0 normalized: mean(x²) = (1+4+9+16)/4 = 7.5; inv = 1/sqrt(7.5).
        let inv0 = (7.5_f32 + eps as f32).sqrt().recip();
        for j in 0..4 {
            let expected = [1.0_f32, 2.0, 3.0, 4.0][j] * inv0;
            assert!(
                (logits[j] - expected).abs() < 1e-5,
                "token 0 dim {}: got {}, want {}",
                j, logits[j], expected
            );
        }
        // Expected token 1: all 2.0 → normalized all = 2.0 * (1/sqrt(4 + eps)) = 2.0 * 0.5 = 1.0.
        for j in 0..4 {
            assert!(
                (logits[4 + j] - 1.0).abs() < 1e-5,
                "token 1 dim {}: got {}",
                j, logits[4 + j]
            );
        }
    }

    /// Output head determinism: same input → same output bit-for-bit.
    #[test]
    fn output_head_deterministic() {
        let h = 8u32;
        let v = 16u32;
        let seq = 3usize;
        let mut seed = 0x2021_u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
        };
        let hidden: Vec<f32> = (0..(seq * h as usize)).map(|_| rand()).collect();
        let output_norm_w: Vec<f32> = (0..(h as usize)).map(|_| 1.0 + rand() * 0.1).collect();
        let output_weight: Vec<f32> = (0..(v * h) as usize).map(|_| rand()).collect();

        let l1 = apply_output_head(&hidden, &output_norm_w, &output_weight, h, v, 1e-6);
        let l2 = apply_output_head(&hidden, &output_norm_w, &output_weight, h, v, 1e-6);
        for i in 0..l1.len() {
            assert_eq!(l1[i].to_bits(), l2[i].to_bits(), "non-deterministic at {}", i);
        }
    }

    /// Greedy argmax picks the highest logit of the LAST token.
    #[test]
    fn greedy_argmax_picks_highest_last_token() {
        // 2 tokens × 5 vocab.
        let logits: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, // token 0: argmax = 4 (but we ignore)
            7.0, -1.0, 5.0, 10.0, 3.0, // token 1: argmax = 3
        ];
        let picked = greedy_argmax_last_token(&logits, 5);
        assert_eq!(picked, 3);
    }

    #[test]
    fn greedy_argmax_handles_single_token() {
        let logits = vec![0.1_f32, 0.2, 0.15, 0.19];
        let picked = greedy_argmax_last_token(&logits, 4);
        assert_eq!(picked, 1);
    }
}
