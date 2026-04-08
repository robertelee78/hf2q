//! Tokenizer loading and calibration data preparation.
//!
//! Wraps HuggingFace's `tokenizers` crate for encoding calibration text
//! that feeds the GPU forward pass during DWQ and quality measurement.

use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;
use tracing::debug;

/// Load a tokenizer from a model directory.
///
/// Looks for `tokenizer.json` in the given directory.
pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !tokenizer_path.exists() {
        anyhow::bail!(
            "tokenizer.json not found in {}. Required for calibration/quality measurement.",
            model_dir.display()
        );
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    debug!(
        vocab_size = tokenizer.get_vocab_size(true),
        "Loaded tokenizer"
    );

    Ok(tokenizer)
}

/// Encode text into token IDs using the given tokenizer.
///
/// Returns the token IDs without special tokens (BOS/EOS) so the caller
/// can add them as needed for the specific model architecture.
pub fn encode_calibration_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))
        .context("Failed to encode calibration text")?;

    let ids: Vec<u32> = encoding.get_ids().to_vec();

    debug!(
        text_len = text.len(),
        token_count = ids.len(),
        "Encoded calibration text"
    );

    Ok(ids)
}

/// Default calibration text for when no dataset is specified.
///
/// This is a short, linguistically diverse passage that exercises common
/// token patterns. It covers technical writing, natural language, and
/// punctuation variety to produce a representative activation distribution.
pub fn default_calibration_text() -> &'static str {
    r#"The quick brown fox jumps over the lazy dog. In machine learning, neural networks are composed of layers that transform input data through learned parameters. Each layer applies a linear transformation followed by a non-linear activation function such as ReLU, GELU, or SiLU.

Transformer architectures use self-attention mechanisms to capture long-range dependencies in sequences. The attention score between query Q and key K is computed as softmax(QK^T / sqrt(d_k)) * V, where d_k is the dimension of the key vectors.

Modern large language models like LLaMA, Mistral, and Gemma employ techniques such as:
1. Rotary Position Embeddings (RoPE) for encoding positional information
2. Grouped Query Attention (GQA) to reduce memory bandwidth requirements
3. SwiGLU activation functions in the feed-forward network
4. RMSNorm for pre-normalization of each sub-layer

Quantization reduces model size by representing weights with fewer bits. Common methods include round-to-nearest (RTN), GPTQ, AWQ, and dynamic weight quantization (DWQ). The key challenge is minimizing quality degradation while maximizing compression ratio.

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(f"The 10th Fibonacci number is {fibonacci(10)}")

SELECT users.name, COUNT(orders.id) AS order_count
FROM users LEFT JOIN orders ON users.id = orders.user_id
GROUP BY users.name HAVING order_count > 5
ORDER BY order_count DESC LIMIT 10;

{"model": "llama-3.1-8b", "quantization": "q4_k_m", "perplexity": 5.92, "tokens_per_second": 84.3}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_calibration_text_not_empty() {
        let text = default_calibration_text();
        assert!(!text.is_empty());
        // Should contain representative patterns
        assert!(text.contains("Transformer"));
        assert!(text.contains("quantization") || text.contains("Quantization"));
        assert!(text.contains("fibonacci") || text.contains("Fibonacci"));
    }

    #[test]
    fn test_default_calibration_text_reasonable_length() {
        let text = default_calibration_text();
        // Should be substantial enough for calibration (at least 500 chars)
        assert!(text.len() > 500);
        // But not excessively long (under 5000 chars)
        assert!(text.len() < 5000);
    }

    #[test]
    fn test_load_tokenizer_missing_file() {
        let result = load_tokenizer(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("tokenizer.json not found"));
    }
}
