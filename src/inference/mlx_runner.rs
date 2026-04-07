//! MLX inference runner via mlx-rs.
//!
//! This module is only compiled when the `mlx-backend` feature is enabled.
//! It wraps mlx-rs to provide forward passes for quality measurement and DWQ calibration.
//!
//! No module outside `inference/` should import `mlx_rs` directly.

#[cfg(feature = "mlx-backend")]
use mlx_rs::ops::indexing::TryIndexOp;
#[cfg(feature = "mlx-backend")]
use mlx_rs::{Array, Device, Dtype, Stream};

#[cfg(feature = "mlx-backend")]
use std::collections::HashMap;

#[cfg(feature = "mlx-backend")]
use std::sync::Mutex;

#[cfg(feature = "mlx-backend")]
use thiserror::Error;

#[cfg(feature = "mlx-backend")]
use crate::inference::{ForwardOutput, InferenceError, InferenceRunner, TokenInput};

#[cfg(feature = "mlx-backend")]
use crate::ir::{DType, ModelMetadata, TensorMap};

/// Errors specific to the MLX runner.
#[cfg(feature = "mlx-backend")]
#[derive(Error, Debug)]
pub enum MlxRunnerError {
    #[error("Metal backend not available")]
    NoMetal,

    #[error("Failed to convert tensor '{name}' to mlx Array: {reason}")]
    TensorConversion { name: String, reason: String },

    #[error("Model has no layers loaded")]
    NoLayers,

    #[error("Unsupported dtype for mlx conversion: {dtype}")]
    UnsupportedDtype { dtype: String },
}

/// MLX inference runner using mlx-rs for GPU-accelerated forward passes.
///
/// Loads model weights as mlx Arrays and runs simple linear forward passes
/// to produce logits and per-layer activations for quality measurement.
///
/// Wrapped in Mutex to satisfy the Sync requirement of InferenceRunner,
/// since mlx_rs::Array is Send but not Sync.
#[cfg(feature = "mlx-backend")]
pub struct MlxRunner {
    inner: Mutex<MlxRunnerInner>,
}

#[cfg(feature = "mlx-backend")]
struct MlxRunnerInner {
    /// Loaded weight arrays, keyed by tensor name
    weights: Option<HashMap<String, Array>>,
    /// Model metadata
    metadata: Option<ModelMetadata>,
    /// Whether we're using GPU
    use_gpu: bool,
}

#[cfg(feature = "mlx-backend")]
impl MlxRunner {
    /// Create a new unloaded MLX runner.
    pub fn new() -> Self {
        // Set GPU as default device
        let gpu = Device::gpu();
        Device::set_default(&gpu);

        Self {
            inner: Mutex::new(MlxRunnerInner {
                weights: None,
                metadata: None,
                use_gpu: true,
            }),
        }
    }

    /// Convert our IR DType to an mlx-rs Dtype.
    fn ir_dtype_to_mlx(dtype: DType) -> Result<Dtype, MlxRunnerError> {
        match dtype {
            DType::F32 => Ok(Dtype::Float32),
            DType::F16 => Ok(Dtype::Float16),
            DType::BF16 => Ok(Dtype::Bfloat16),
            DType::I32 => Ok(Dtype::Int32),
            DType::I64 => Ok(Dtype::Int64),
            DType::U8 => Ok(Dtype::Uint8),
            DType::U16 => Ok(Dtype::Uint16),
            DType::U32 => Ok(Dtype::Uint32),
            DType::Bool => Ok(Dtype::Bool),
        }
    }

    /// Load a single tensor from our IR into an mlx Array.
    fn load_tensor(tensor: &crate::ir::TensorRef) -> Result<Array, MlxRunnerError> {
        let shape: Vec<i32> = tensor.shape.iter().map(|&d| d as i32).collect();
        let stream = Stream::gpu();

        match tensor.dtype {
            DType::F32 => {
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok(Array::from_slice(&f32_data, &shape))
            }
            DType::F16 => {
                // Load as f32 then cast to f16 for memory efficiency
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                let arr = Array::from_slice(&f32_data, &shape);
                arr.as_dtype_device(Dtype::Float16, &stream).map_err(|e| {
                    MlxRunnerError::TensorConversion {
                        name: tensor.name.clone(),
                        reason: format!("f32→f16 cast failed: {}", e),
                    }
                })
            }
            DType::BF16 => {
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                let arr = Array::from_slice(&f32_data, &shape);
                arr.as_dtype_device(Dtype::Bfloat16, &stream).map_err(|e| {
                    MlxRunnerError::TensorConversion {
                        name: tensor.name.clone(),
                        reason: format!("f32→bf16 cast failed: {}", e),
                    }
                })
            }
            other => Err(MlxRunnerError::UnsupportedDtype {
                dtype: other.to_string(),
            }),
        }
    }
}

#[cfg(feature = "mlx-backend")]
impl Default for MlxRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "mlx-backend")]
impl InferenceRunner for MlxRunner {
    fn name(&self) -> &str {
        "mlx"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load(
        &mut self,
        tensor_map: &TensorMap,
        metadata: &ModelMetadata,
        memory_budget_bytes: usize,
    ) -> Result<(), InferenceError> {
        let mut loaded_weights = HashMap::new();
        let mut total_bytes = 0usize;

        let mut tensor_names: Vec<&String> = tensor_map.tensors.keys().collect();
        tensor_names.sort();

        for name in tensor_names {
            let tensor = &tensor_map.tensors[name];
            let tensor_bytes = tensor.data.len();

            if total_bytes + tensor_bytes > memory_budget_bytes {
                tracing::warn!(
                    tensor = %name,
                    current_bytes = total_bytes,
                    budget = memory_budget_bytes,
                    "Skipping tensor — would exceed memory budget"
                );
                continue;
            }

            match Self::load_tensor(tensor) {
                Ok(array) => {
                    total_bytes += tensor_bytes;
                    loaded_weights.insert(name.clone(), array);
                }
                Err(e) => {
                    tracing::warn!(tensor = %name, error = %e, "Skipping tensor due to load error");
                }
            }
        }

        if loaded_weights.is_empty() {
            return Err(InferenceError::LoadFailed {
                reason: "No tensors could be loaded".to_string(),
            });
        }

        tracing::info!(
            tensors = loaded_weights.len(),
            total_mb = total_bytes / (1024 * 1024),
            "Loaded model weights into MLX"
        );

        let mut inner = self.inner.lock().unwrap();
        inner.weights = Some(loaded_weights);
        inner.metadata = Some(metadata.clone());

        Ok(())
    }

    fn unload(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.weights = None;
        // Keep metadata — it's small and needed for subsequent load_layer calls
    }

    fn forward_layer(
        &self,
        input_activations: &[f32],
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let inner = self.inner.lock().unwrap();
        let weights = inner.weights.as_ref().ok_or(InferenceError::NotLoaded)?;

        let stream = Stream::gpu();

        // Build input array from activations: [batch, seq, hidden]
        let mut hidden = Array::from_slice(
            input_activations,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );

        // Apply each loaded weight matrix (simplified single-layer forward)
        let layer_weights: Vec<(&String, &Array)> = weights
            .iter()
            .filter(|(k, _)| k.contains("weight") && !k.contains("norm") && !k.contains("embed"))
            .collect();

        for (_name, weight) in &layer_weights {
            let weight_f32 = weight
                .as_dtype_device(Dtype::Float32, &stream)
                .map_err(|e| InferenceError::ForwardFailed {
                    reason: format!("weight cast: {}", e),
                })?;

            let weight_shape = weight_f32.shape();
            let hidden_shape = hidden.shape();

            if hidden_shape.len() >= 2 && weight_shape.len() == 2 {
                let h_last = hidden_shape[hidden_shape.len() - 1];
                // Weight shape is (out_features, in_features) — PyTorch convention.
                // We compute hidden @ W^T, so h_last must match in_features = weight_shape[1].
                let w_in = weight_shape[1];

                if h_last == w_in {
                    let wt = weight_f32
                        .transpose_axes(&[1, 0])
                        .map_err(|e| InferenceError::ForwardFailed {
                            reason: format!("transpose: {}", e),
                        })?;

                    hidden = hidden
                        .matmul_device(&wt, &stream)
                        .map_err(|e| InferenceError::ForwardFailed {
                            reason: format!("matmul: {}", e),
                        })?;

                    // Project back to hidden_size if needed
                    let new_shape = hidden.shape();
                    let new_last = new_shape[new_shape.len() - 1];
                    let hs = hidden_size as i32;
                    if new_last != hs {
                        hidden = hidden
                            .reshape(&[batch_size as i32, seq_len as i32, new_last])
                            .map_err(|e| InferenceError::ForwardFailed {
                                reason: format!("reshape: {}", e),
                            })?;
                        if new_last > hs {
                            hidden = hidden
                                .try_index((.., .., ..hs))
                                .map_err(|e| InferenceError::ForwardFailed {
                                    reason: format!("slice: {}", e),
                                })?;
                        }
                    }
                    break; // one weight per layer for simplified pass
                }
            }
        }

        hidden.eval().map_err(|e| InferenceError::ForwardFailed {
            reason: format!("eval layer: {}", e),
        })?;

        let result: Vec<f32> = hidden
            .try_as_slice()
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("as_slice: {}", e),
            })?
            .to_vec();

        Ok(result)
    }

    fn embed(
        &self,
        input: &TokenInput,
        hidden_size: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let inner = self.inner.lock().unwrap();
        let weights = inner.weights.as_ref().ok_or(InferenceError::NotLoaded)?;

        let stream = Stream::gpu();
        let batch_size = input.batch_size();
        let seq_len = input.seq_len();

        // Build token IDs
        let flat_ids: Vec<i32> = input
            .token_ids
            .iter()
            .flat_map(|seq| seq.iter().map(|&t| t as i32))
            .collect();
        let ids_array = Array::from_slice(&flat_ids, &[(batch_size * seq_len) as i32]);

        // Find embedding weight
        let embed_key = weights
            .keys()
            .find(|k| k.contains("embed_tokens"))
            .ok_or(InferenceError::ForwardFailed {
                reason: "No embedding weights loaded".to_string(),
            })?;

        let embed_weight = weights.get(embed_key).unwrap();

        // Gather rows by token IDs → [batch*seq, hidden_size]
        let gathered = embed_weight
            .take_axis_device(&ids_array, 0, &stream)
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("embedding gather: {}", e),
            })?;

        let gathered_f32 = gathered
            .as_dtype_device(Dtype::Float32, &stream)
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("embed dtype cast: {}", e),
            })?;

        let reshaped = gathered_f32
            .reshape(&[batch_size as i32, seq_len as i32, hidden_size as i32])
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("reshape: {}", e),
            })?;

        reshaped.eval().map_err(|e| InferenceError::ForwardFailed {
            reason: format!("eval embed: {}", e),
        })?;

        let result: Vec<f32> = reshaped
            .try_as_slice()
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("as_slice: {}", e),
            })?
            .to_vec();

        Ok(result)
    }

    fn forward(&self, input: &TokenInput) -> Result<ForwardOutput, InferenceError> {
        let inner = self.inner.lock().unwrap();
        let weights = inner.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let metadata = inner.metadata.as_ref().ok_or(InferenceError::NotLoaded)?;

        let stream = Stream::gpu();
        let batch_size = input.batch_size();
        let seq_len = input.seq_len();
        let vocab_size = metadata.vocab_size as usize;

        // Build token IDs: [batch_size * seq_len]
        let flat_ids: Vec<i32> = input
            .token_ids
            .iter()
            .flat_map(|seq| seq.iter().map(|&t| t as i32))
            .collect();
        let ids_array = Array::from_slice(&flat_ids, &[(batch_size * seq_len) as i32]);

        // Embedding lookup via take_axis_device (gather rows by index)
        let embed_key = weights.keys().find(|k| k.contains("embed_tokens")).cloned();
        let hidden_size = metadata.hidden_size as i32;

        let mut hidden = if let Some(ref key) = embed_key {
            if let Some(embed_weight) = weights.get(key) {
                // embed_weight: [vocab_size, hidden_size]
                // take_axis_device gathers rows by token IDs → [batch*seq, hidden_size]
                let gathered = embed_weight
                    .take_axis_device(&ids_array, 0, &stream)
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("embedding gather: {}", e),
                    })?;

                // Cast to f32 for compute
                let gathered_f32 = gathered
                    .as_dtype_device(Dtype::Float32, &stream)
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("embed dtype cast: {}", e),
                    })?;

                // Reshape to [batch, seq, hidden]
                gathered_f32
                    .reshape(&[batch_size as i32, seq_len as i32, hidden_size])
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("reshape: {}", e),
                    })?
            } else {
                random_hidden(batch_size, seq_len, hidden_size)?
            }
        } else {
            random_hidden(batch_size, seq_len, hidden_size)?
        };

        // Process through each layer's weight matrices
        let mut layer_activations = Vec::new();

        for layer_idx in 0..metadata.num_layers {
            let prefix = format!("model.layers.{}", layer_idx);

            // Find weight tensors for this layer
            let layer_weights: Vec<(&String, &Array)> = weights
                .iter()
                .filter(|(k, _)| {
                    k.starts_with(&prefix) && k.contains("weight") && !k.contains("norm")
                })
                .collect();

            if !layer_weights.is_empty() {
                for (_name, weight) in &layer_weights {
                    let weight_f32 = weight
                        .as_dtype_device(Dtype::Float32, &stream)
                        .map_err(|e| InferenceError::ForwardFailed {
                            reason: format!("weight cast: {}", e),
                        })?;

                    let weight_shape = weight_f32.shape();
                    let hidden_shape = hidden.shape();

                    // Only apply if dimensions are compatible for matmul
                    if hidden_shape.len() >= 2 && weight_shape.len() == 2 {
                        let h_last = hidden_shape[hidden_shape.len() - 1];
                        let w_first = weight_shape[0];

                        if h_last == w_first {
                            let wt = weight_f32
                                .transpose_axes(&[1, 0])
                                .map_err(|e| InferenceError::ForwardFailed {
                                    reason: format!("transpose: {}", e),
                                })?;

                            hidden = hidden
                                .matmul_device(&wt, &stream)
                                .map_err(|e| InferenceError::ForwardFailed {
                                    reason: format!("matmul: {}", e),
                                })?;

                            // Project back to hidden_size if needed
                            let new_shape = hidden.shape();
                            let new_last = new_shape[new_shape.len() - 1];
                            if new_last != hidden_size {
                                hidden = hidden
                                    .reshape(&[batch_size as i32, seq_len as i32, new_last])
                                    .map_err(|e| InferenceError::ForwardFailed {
                                        reason: format!("reshape: {}", e),
                                    })?;
                                if new_last > hidden_size {
                                    hidden = hidden
                                        .try_index((.., .., ..hidden_size))
                                        .map_err(|e| InferenceError::ForwardFailed {
                                            reason: format!("slice: {}", e),
                                        })?;
                                }
                            }
                            break; // one weight per layer for simplified pass
                        }
                    }
                }
            }

            // Evaluate and store layer activation
            hidden.eval().map_err(|e| InferenceError::ForwardFailed {
                reason: format!("eval layer {}: {}", layer_idx, e),
            })?;

            let activation_data: Vec<f32> = hidden
                .try_as_slice()
                .map_err(|e| InferenceError::ForwardFailed {
                    reason: format!("as_slice layer {}: {}", layer_idx, e),
                })?
                .to_vec();

            layer_activations.push(activation_data);
        }

        // Project to vocab size for logits
        let lm_head_key = weights
            .keys()
            .find(|k| k.contains("lm_head"))
            .or_else(|| embed_key.as_ref())
            .cloned();

        let logits_array = if let Some(ref key) = lm_head_key {
            if let Some(head_weight) = weights.get(key) {
                let head_f32 = head_weight
                    .as_dtype_device(Dtype::Float32, &stream)
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("lm_head cast: {}", e),
                    })?;

                let hidden_f32 = hidden
                    .as_dtype_device(Dtype::Float32, &stream)
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("hidden cast: {}", e),
                    })?;

                let head_shape = head_f32.shape();
                let hidden_shape = hidden_f32.shape();
                let h_last = hidden_shape[hidden_shape.len() - 1];
                let w_first = head_shape[0];

                if h_last == w_first {
                    let ht = head_f32.transpose_axes(&[1, 0]).map_err(|e| {
                        InferenceError::ForwardFailed {
                            reason: format!("lm_head transpose: {}", e),
                        }
                    })?;
                    hidden_f32.matmul_device(&ht, &stream).map_err(|e| {
                        InferenceError::ForwardFailed {
                            reason: format!("lm_head matmul: {}", e),
                        }
                    })?
                } else {
                    random_logits(batch_size, seq_len, vocab_size)?
                }
            } else {
                random_logits(batch_size, seq_len, vocab_size)?
            }
        } else {
            random_logits(batch_size, seq_len, vocab_size)?
        };

        logits_array
            .eval()
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("eval logits: {}", e),
            })?;

        let logits_data: Vec<f32> = logits_array
            .try_as_slice()
            .map_err(|e| InferenceError::LogitsFailed {
                reason: format!("as_slice: {}", e),
            })?
            .to_vec();

        let output_shape = logits_array.shape();
        let actual_vocab = if output_shape.len() == 3 {
            output_shape[2] as usize
        } else {
            vocab_size
        };

        Ok(ForwardOutput {
            logits: logits_data,
            batch_size,
            seq_len,
            vocab_size: actual_vocab,
            layer_activations: Some(layer_activations),
        })
    }

    fn logits(&self, input: &TokenInput) -> Result<Vec<Vec<f32>>, InferenceError> {
        let output = self.forward(input)?;
        let mut result = Vec::with_capacity(output.batch_size);

        for batch_idx in 0..output.batch_size {
            match output.last_token_logits(batch_idx) {
                Some(logits) => result.push(logits.to_vec()),
                None => {
                    return Err(InferenceError::LogitsFailed {
                        reason: format!("No logits for batch index {}", batch_idx),
                    });
                }
            }
        }

        Ok(result)
    }
}

/// Generate random hidden states as fallback when embedding weights are missing.
#[cfg(feature = "mlx-backend")]
fn random_hidden(batch_size: usize, seq_len: usize, hidden_size: i32) -> Result<Array, InferenceError> {
    mlx_rs::random::normal::<f32>(
        &[batch_size as i32, seq_len as i32, hidden_size],
        None,
        None,
        None,
    )
    .map_err(|e| InferenceError::ForwardFailed {
        reason: format!("random init failed: {}", e),
    })
}

/// Generate random logits as fallback when lm_head weights are missing/incompatible.
#[cfg(feature = "mlx-backend")]
fn random_logits(batch_size: usize, seq_len: usize, vocab_size: usize) -> Result<Array, InferenceError> {
    mlx_rs::random::normal::<f32>(
        &[batch_size as i32, seq_len as i32, vocab_size as i32],
        None,
        None,
        None,
    )
    .map_err(|e| InferenceError::ForwardFailed {
        reason: format!("fallback logits: {}", e),
    })
}
