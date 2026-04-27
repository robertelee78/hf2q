//! Embedding pooling — pure-Rust reference implementations and strategy
//! dispatch rules (ADR-005 Phase 2b, T1.9 wave-2).
//!
//! # Background
//!
//! An embedding model's forward pass produces a `[seq_len, hidden]` matrix of
//! per-token hidden states. A *pooling strategy* reduces this to a single
//! `[hidden]` vector that represents the entire input sequence — this is the
//! vector returned by `POST /v1/embeddings`.
//!
//! # Strategy → Model dispatch rules
//!
//! | Strategy          | Representative models                                     | GGUF `pooling_type` |
//! |-------------------|-----------------------------------------------------------|---------------------|
//! | [`Mean`]          | nomic-embed-text-v1.5, BGE-M3 dense head, mxbai-embed    | `1`                 |
//! | [`ClsToken`]      | bge-small-en-v1.5, classic BERT-base                      | `2`                 |
//! | [`EosLastToken`]  | Qwen3-Embedding, E5-mistral-7b-instruct (causal decoders) | `3` (`Last`)        |
//!
//! **`EosLastToken` semantics:** for causal decoder models used as embedders
//! (Qwen3-Embedding, E5-mistral), the special EOS token is appended to the
//! input and the hidden state of that last real token (position
//! `valid_token_count - 1`) is taken as the pooled representation.
//! Mathematically this is identical to `Last` pooling when applied with
//! `valid_token_count` rather than padded `seq_len`.  These models carry
//! `pooling_type = 3` in their GGUF metadata; the GPU forward pass in
//! [`crate::inference::models::bert::bert_gpu::apply_bert_full_forward_gpu`]
//! and
//! [`crate::inference::models::nomic_bert::apply_nomic_bert_full_forward_gpu`]
//! dispatch this automatically via
//! [`crate::inference::models::bert::config::PoolingType::Last`] →
//! [`crate::inference::models::bert::bert_gpu::BertPoolKind::Last`].
//!
//! # CPU reference implementations
//!
//! [`pool_mean`] and [`pool_eos_last_token`] are pure-Rust, GPU-free
//! implementations of the two most common strategies.  They serve two
//! purposes:
//!   1. **Unit-testable documentation** — the semantics are verified without a
//!      Metal device or kernel registry.
//!   2. **Parity reference** — new GPU kernels can be validated against these
//!      before shipping.
//!
//! Production embeddings run through the GPU path (Metal kernels via
//! mlx-native).  The CPU path is never invoked in the hot path.

use crate::inference::models::bert::config::PoolingType;

/// Pooling strategy for reducing `[seq_len, hidden]` → `[hidden]`.
///
/// Determined from the model's GGUF metadata (`{arch}.pooling_type` key).
/// See the module-level doc for the dispatch rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average all non-padding token hidden states.
    ///
    /// `out[d] = sum_{i=0}^{valid-1} h[i,d] / valid`
    ///
    /// Used by: nomic-embed-text-v1.5, BGE-M3 dense, mxbai-embed-large.
    Mean,

    /// Take the hidden state of the CLS token (position 0).
    ///
    /// `out[d] = h[0, d]`
    ///
    /// Used by: bge-small-en-v1.5, classic BERT-base.
    ClsToken,

    /// Take the hidden state of the last non-padding token
    /// (position `valid_token_count - 1`).
    ///
    /// For causal-decoder embedding models (Qwen3-Embedding,
    /// E5-mistral-7b-instruct) the input is constructed so that the EOS
    /// token falls at this position, giving the name "EOS-last-token".
    ///
    /// Used by: Qwen3-Embedding family, E5-mistral-7b-instruct.
    EosLastToken,
}

impl PoolingStrategy {
    /// Convert from the GGUF `pooling_type` numeric value.
    ///
    /// Mapping mirrors `llama_pooling_type` in llama.cpp:
    /// - `1` → [`Mean`](PoolingStrategy::Mean)
    /// - `2` → [`ClsToken`](PoolingStrategy::ClsToken)
    /// - `3` → [`EosLastToken`](PoolingStrategy::EosLastToken) (llama.cpp `Last`)
    ///
    /// Returns `None` for unrecognised values (e.g. `0` = None,
    /// `4` = Rank — both unsupported by `/v1/embeddings`).
    pub fn from_pooling_type(pt: PoolingType) -> Option<Self> {
        match pt {
            PoolingType::Mean => Some(Self::Mean),
            PoolingType::Cls => Some(Self::ClsToken),
            PoolingType::Last => Some(Self::EosLastToken),
            PoolingType::None | PoolingType::Rank => None,
        }
    }

    /// Human-readable name — stable for logging / ADR doc references.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::ClsToken => "cls",
            Self::EosLastToken => "eos_last_token",
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference pooling functions
// ---------------------------------------------------------------------------

/// Mean-pool a `[seq_len × hidden]` row-major buffer over the first
/// `valid_token_count` rows.
///
/// Returns a `Vec<f32>` of length `hidden`.
///
/// # Panics
/// - `valid_token_count == 0`
/// - `valid_token_count > seq_len`
/// - `hidden_states.len() != seq_len * hidden`
/// - `hidden == 0`
pub fn pool_mean(
    hidden_states: &[f32],
    seq_len: usize,
    hidden: usize,
    valid_token_count: usize,
) -> Vec<f32> {
    assert!(hidden > 0, "pool_mean: hidden must be > 0");
    assert!(
        valid_token_count > 0,
        "pool_mean: valid_token_count must be > 0"
    );
    assert!(
        valid_token_count <= seq_len,
        "pool_mean: valid_token_count ({}) > seq_len ({})",
        valid_token_count,
        seq_len
    );
    assert_eq!(
        hidden_states.len(),
        seq_len * hidden,
        "pool_mean: hidden_states length {} != seq_len {} × hidden {}",
        hidden_states.len(),
        seq_len,
        hidden
    );

    let mut out = vec![0.0f32; hidden];
    for row in 0..valid_token_count {
        let base = row * hidden;
        for (d, v) in out.iter_mut().enumerate() {
            *v += hidden_states[base + d];
        }
    }
    let inv = 1.0 / valid_token_count as f32;
    for v in &mut out {
        *v *= inv;
    }
    out
}

/// EOS-last-token pool: take the hidden state of row `valid_token_count - 1`.
///
/// For causal decoder-based embedding models (Qwen3-Embedding,
/// E5-mistral-7b-instruct), the EOS token is placed at this position, making
/// its hidden state a function of the entire preceding context.
///
/// Returns a `Vec<f32>` of length `hidden`.
///
/// # Panics
/// - `valid_token_count == 0`
/// - `valid_token_count > seq_len`
/// - `hidden_states.len() != seq_len * hidden`
/// - `hidden == 0`
pub fn pool_eos_last_token(
    hidden_states: &[f32],
    seq_len: usize,
    hidden: usize,
    valid_token_count: usize,
) -> Vec<f32> {
    assert!(hidden > 0, "pool_eos_last_token: hidden must be > 0");
    assert!(
        valid_token_count > 0,
        "pool_eos_last_token: valid_token_count must be > 0"
    );
    assert!(
        valid_token_count <= seq_len,
        "pool_eos_last_token: valid_token_count ({}) > seq_len ({})",
        valid_token_count,
        seq_len
    );
    assert_eq!(
        hidden_states.len(),
        seq_len * hidden,
        "pool_eos_last_token: hidden_states length {} != seq_len {} × hidden {}",
        hidden_states.len(),
        seq_len,
        hidden
    );

    let last_row = valid_token_count - 1;
    let base = last_row * hidden;
    hidden_states[base..base + hidden].to_vec()
}

/// L2-normalize a vector in-place: `v[i] /= sqrt(sum(v[i]^2) + eps)`.
///
/// Used as the final step of the embedding pipeline (after pooling) to produce
/// unit-norm vectors suitable for cosine-similarity search.
///
/// `eps` prevents division by zero on all-zero inputs; standard value is
/// `1e-12` (matches llama.cpp + sentence-transformers).
pub fn l2_normalize(v: &mut [f32], eps: f32) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let inv_norm = 1.0 / (norm_sq + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // pool_mean
    // -----------------------------------------------------------------------

    /// Toy 3-token, 2-dim input, all tokens valid.
    ///
    /// hidden_states (row-major, 3 × 2):
    ///   row 0: [1.0, 2.0]
    ///   row 1: [3.0, 4.0]
    ///   row 2: [5.0, 6.0]
    ///
    /// expected mean over all 3 rows: [(1+3+5)/3, (2+4+6)/3] = [3.0, 4.0]
    #[test]
    fn pool_mean_all_tokens_valid() {
        let hs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = pool_mean(&hs, 3, 2, 3);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.0).abs() < 1e-6, "dim 0: got {}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-6, "dim 1: got {}", out[1]);
    }

    /// 3-token, 2-dim input with only 2 valid tokens (1 pad).
    ///
    /// hidden_states:
    ///   row 0: [1.0, 2.0]
    ///   row 1: [3.0, 4.0]
    ///   row 2: [99.0, 99.0]  ← padding — must NOT contribute
    ///
    /// expected mean over rows 0..1: [(1+3)/2, (2+4)/2] = [2.0, 3.0]
    #[test]
    fn pool_mean_masks_padding_rows() {
        let hs = vec![1.0f32, 2.0, 3.0, 4.0, 99.0, 99.0];
        let out = pool_mean(&hs, 3, 2, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 2.0).abs() < 1e-6, "dim 0: got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-6, "dim 1: got {}", out[1]);
    }

    /// Single valid token — mean of one row is that row.
    #[test]
    fn pool_mean_single_token() {
        let hs = vec![7.0f32, -3.5, 0.0, 2.0];
        let out = pool_mean(&hs, 1, 4, 1);
        assert_eq!(out, vec![7.0, -3.5, 0.0, 2.0]);
    }

    // -----------------------------------------------------------------------
    // pool_eos_last_token
    // -----------------------------------------------------------------------

    /// 3-token, 2-dim, valid_token_count = 3 → picks row 2.
    ///
    /// hidden_states:
    ///   row 0: [0.1, 0.2]
    ///   row 1: [0.3, 0.4]
    ///   row 2: [0.5, 0.6]  ← EOS position
    #[test]
    fn pool_eos_last_picks_last_valid_row() {
        let hs = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = pool_eos_last_token(&hs, 3, 2, 3);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-7, "dim 0: got {}", out[0]);
        assert!((out[1] - 0.6).abs() < 1e-7, "dim 1: got {}", out[1]);
    }

    /// Padded case: 3 total tokens, 2 valid → row 1 is the EOS position;
    /// row 2 is padding and must NOT be selected.
    #[test]
    fn pool_eos_last_ignores_padding() {
        let hs = vec![0.1f32, 0.2, 0.5, 0.6, 99.0, 99.0];
        let out = pool_eos_last_token(&hs, 3, 2, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-7, "dim 0: got {}", out[0]);
        assert!((out[1] - 0.6).abs() < 1e-7, "dim 1: got {}", out[1]);
    }

    /// Single valid token.
    #[test]
    fn pool_eos_last_single_token() {
        let hs = vec![3.0f32, -1.0, 2.5];
        let out = pool_eos_last_token(&hs, 1, 3, 1);
        assert_eq!(out, vec![3.0, -1.0, 2.5]);
    }

    // -----------------------------------------------------------------------
    // l2_normalize
    // -----------------------------------------------------------------------

    /// Known unit vector should stay unit.
    #[test]
    fn l2_normalize_already_unit() {
        let mut v = vec![1.0f32, 0.0, 0.0];
        l2_normalize(&mut v, 1e-12);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!(v[2].abs() < 1e-6);
    }

    /// After normalizing [3, 4], norm should be 1.0 (3-4-5 triple / 5).
    #[test]
    fn l2_normalize_three_four() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v, 1e-12);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm = {}", norm);
        assert!((v[0] - 0.6).abs() < 1e-6, "v[0] = {}", v[0]);
        assert!((v[1] - 0.8).abs() < 1e-6, "v[1] = {}", v[1]);
    }

    /// All-zeros input should not panic (eps guards division by zero).
    #[test]
    fn l2_normalize_all_zeros_no_panic() {
        let mut v = vec![0.0f32; 8];
        l2_normalize(&mut v, 1e-12);
        // All values stay near 0; no panic.
        for x in &v {
            assert!(x.is_finite(), "non-finite after normalizing zeros");
        }
    }

    // -----------------------------------------------------------------------
    // PoolingStrategy
    // -----------------------------------------------------------------------

    #[test]
    fn pooling_strategy_from_pooling_type_round_trips() {
        use crate::inference::models::bert::config::PoolingType;
        assert_eq!(
            PoolingStrategy::from_pooling_type(PoolingType::Mean),
            Some(PoolingStrategy::Mean)
        );
        assert_eq!(
            PoolingStrategy::from_pooling_type(PoolingType::Cls),
            Some(PoolingStrategy::ClsToken)
        );
        assert_eq!(
            PoolingStrategy::from_pooling_type(PoolingType::Last),
            Some(PoolingStrategy::EosLastToken)
        );
        assert_eq!(
            PoolingStrategy::from_pooling_type(PoolingType::None),
            None
        );
        assert_eq!(
            PoolingStrategy::from_pooling_type(PoolingType::Rank),
            None
        );
    }

    #[test]
    fn pooling_strategy_as_str_stable() {
        assert_eq!(PoolingStrategy::Mean.as_str(), "mean");
        assert_eq!(PoolingStrategy::ClsToken.as_str(), "cls");
        assert_eq!(PoolingStrategy::EosLastToken.as_str(), "eos_last_token");
    }

    // -----------------------------------------------------------------------
    // Dispatch consistency check
    // -----------------------------------------------------------------------

    /// Verify that pool_mean and pool_eos_last_token agree on a 1-token
    /// sequence (both should return the single row).
    #[test]
    fn pool_mean_and_eos_last_agree_on_single_token() {
        let hs = vec![1.0f32, 2.0, 3.0, 4.0];
        let mean_out = pool_mean(&hs, 1, 4, 1);
        let last_out = pool_eos_last_token(&hs, 1, 4, 1);
        assert_eq!(mean_out, last_out);
    }
}
