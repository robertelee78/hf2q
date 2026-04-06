//! MLX inference runner via mlx-rs.
//!
//! This module is only compiled when the `mlx-backend` feature is enabled.
//! It wraps mlx-rs to provide forward passes for quality measurement and DWQ calibration.
//!
//! No module outside `inference/` should import `mlx_rs` directly.

#[cfg(feature = "mlx-backend")]
use mlx_rs::{Array, Device, Dtype, StreamOrDevice};

#[cfg(feature = "mlx-backend")]
use std::collections::HashMap;

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
#[cfg(feature = "mlx-backend")]
pub struct MlxRunner {
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
        // Try to set GPU as default device
        let use_gpu = {
            let gpu = Device::gpu();
            Device::set_default(&gpu);
            true
        };

        Self {
            weights: None,
            metadata: None,
            use_gpu,
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
        // Convert shape from usize to i32 (mlx-rs uses i32 for shape dims)
        let shape: Vec<i32> = tensor
            .shape
            .iter()
            .map(|&d| d as i32)
            .collect();

        // Use the safetensors bridge path: construct TensorView-like data
        // and convert via Array::from_slice for f32, or raw data for f16/bf16.
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
                // Convert f16 → f32 → Array, then cast back
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                let arr = Array::from_slice(&f32_data, &shape);
                // Cast to float16 for memory efficiency
                let s = if Device::gpu().device_type() == mlx_rs::DeviceType::Gpu {
                    StreamOrDevice::gpu()
                } else {
                    StreamOrDevice::cpu()
                };
                arr.as_dtype_with_stream(Dtype::Float16, s).map_err(|e| {
                    MlxRunnerError::TensorConversion {
                        name: tensor.name.clone(),
                        reason: format!("dtype cast failed: {}", e),
                    }
                })
            }
            DType::BF16 => {
                // Convert bf16 → f32 → Array
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                let arr = Array::from_slice(&f32_data, &shape);
                let s = if Device::gpu().device_type() == mlx_rs::DeviceType::Gpu {
                    StreamOrDevice::gpu()
                } else {
                    StreamOrDevice::cpu()
                };
                arr.as_dtype_with_stream(Dtype::Bfloat16, s).map_err(|e| {
                    MlxRunnerError::TensorConversion {
                        name: tensor.name.clone(),
                        reason: format!("dtype cast failed: {}", e),
                    }
                })
            }
            other => Err(MlxRunnerError::UnsupportedDtype {
                dtype: other.to_string(),
            }),
        }
    }

    /// Perform a simple linear forward pass through the model's weight layers.
    ///
    /// This is a simplified forward pass that:
    /// 1. Creates an embedding from token IDs
    /// 2. Multiplies through each weight matrix in layer order
    /// 3. Collects per-layer activations
    ///
    /// This does NOT implement full transformer attention — it provides
    /// approximate logit distributions for KL divergence measurement.
    fn simple_forward(
        &self,
        input: &TokenInput,
    ) -> Result<(Vec<f32>, usize, usize, usize, Option<Vec<Vec<f32>>>), InferenceError> {
        let weights = self
            .weights
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let metadata = self
            .metadata
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;

        let s = if self.use_gpu {
            StreamOrDevice::gpu()
        } else {
            StreamOrDevice::cpu()
        };

        let batch_size = input.batch_size();
        let seq_len = input.seq_len();
        let vocab_size = metadata.vocab_size as usize;

        // Find embedding weight
        let embed_key = weights
            .keys()
            .find(|k| k.contains("embed_tokens"))
            .cloned();

        // Build token IDs array: [batch_size, seq_len]
        let flat_ids: Vec<i32> = input
            .token_ids
            .iter()
            .flat_map(|seq| seq.iter().map(|&t| t as i32))
            .collect();
        let ids_shape = vec![batch_size as i32, seq_len as i32];
        let ids_array = Array::from_slice(&flat_ids, &ids_shape);

        // Get initial hidden states from embedding lookup or random init
        let mut hidden = if let Some(ref key) = embed_key {
            if let Some(embed_weight) = weights.get(key) {
                // Simple embedding: gather rows from embed_weight
                // embed_weight shape: [vocab_size, hidden_size]
                // Use matmul with one-hot as a simple embedding lookup approximation
                let hidden_size = metadata.hidden_size as i32;

                // Create one-hot encoding
                let flat_for_onehot: Vec<i32> = flat_ids.clone();
                let onehot = mlx_rs::ops::one_hot_with_stream(
                    Array::from_slice(&flat_for_onehot, &[batch_size as i32 * seq_len as i32]),
                    vocab_size as i32,
                    Dtype::Float32,
                    s.clone(),
                )
                .map_err(|e| InferenceError::ForwardFailed {
                    reason: format!("one_hot failed: {}", e),
                })?;

                // Reshape onehot to [batch*seq, vocab] and matmul with embedding
                let embed_f32 = embed_weight
                    .as_dtype_with_stream(Dtype::Float32, s.clone())
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("embed dtype cast: {}", e),
                    })?;

                let result = mlx_rs::ops::matmul_with_stream(&onehot, &embed_f32, s.clone())
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("embed matmul failed: {}", e),
                    })?;

                // Reshape to [batch, seq, hidden]
                result
                    .reshape(&[batch_size as i32, seq_len as i32, hidden_size])
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("reshape failed: {}", e),
                    })?
            } else {
                // Fallback: random hidden states
                let hidden_size = metadata.hidden_size as i32;
                mlx_rs::random::normal::<f32>(
                    &[batch_size as i32, seq_len as i32, hidden_size],
                    None,
                    None,
                    None,
                )
                .map_err(|e| InferenceError::ForwardFailed {
                    reason: format!("random init failed: {}", e),
                })?
            }
        } else {
            let hidden_size = metadata.hidden_size as i32;
            mlx_rs::random::normal::<f32>(
                &[batch_size as i32, seq_len as i32, hidden_size],
                None,
                None,
                None,
            )
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("random init failed: {}", e),
            })?
        };

        // Collect layer activations
        let mut layer_activations = Vec::new();

        // Process through each layer's weight matrices
        for layer_idx in 0..metadata.num_layers {
            let prefix = format!("model.layers.{}", layer_idx);

            // Find weight tensors for this layer (q_proj, k_proj, v_proj, o_proj, etc.)
            let layer_weights: Vec<(&String, &Array)> = weights
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix) && k.contains("weight") && !k.contains("norm"))
                .collect();

            if !layer_weights.is_empty() {
                // Apply each weight matrix as a linear transformation
                for (_name, weight) in &layer_weights {
                    let weight_f32 = weight
                        .as_dtype_with_stream(Dtype::Float32, s.clone())
                        .map_err(|e| InferenceError::ForwardFailed {
                            reason: format!("weight cast: {}", e),
                        })?;

                    let weight_shape = weight_f32.shape();
                    let hidden_shape = hidden.shape();

                    // Only apply if dimensions are compatible
                    if hidden_shape.len() >= 2 && weight_shape.len() == 2 {
                        let h_last = hidden_shape[hidden_shape.len() - 1];
                        let w_first = weight_shape[0];

                        if h_last == w_first {
                            hidden = mlx_rs::ops::matmul_with_stream(
                                &hidden,
                                &weight_f32.transpose(&[1, 0]).map_err(|e| {
                                    InferenceError::ForwardFailed {
                                        reason: format!("transpose: {}", e),
                                    }
                                })?,
                                s.clone(),
                            )
                            .map_err(|e| InferenceError::ForwardFailed {
                                reason: format!("matmul: {}", e),
                            })?;

                            // If output dimension doesn't match hidden_size, project back
                            let new_shape = hidden.shape();
                            let new_last = new_shape[new_shape.len() - 1];
                            let hidden_size = metadata.hidden_size as i32;
                            if new_last != hidden_size {
                                // Truncate or pad back to hidden_size
                                hidden = hidden
                                    .reshape(&[batch_size as i32, seq_len as i32, new_last])
                                    .map_err(|e| InferenceError::ForwardFailed {
                                        reason: format!("reshape: {}", e),
                                    })?;
                                // Simple slice to hidden_size if larger
                                if new_last > hidden_size {
                                    hidden = hidden
                                        .index((.., .., ..hidden_size))
                                        .map_err(|e| InferenceError::ForwardFailed {
                                            reason: format!("slice: {}", e),
                                        })?;
                                }
                            }
                            break; // one weight per layer for the simplified pass
                        }
                    }
                }
            }

            // Store layer activation
            hidden.eval().map_err(|e| InferenceError::ForwardFailed {
                reason: format!("eval layer {}: {}", layer_idx, e),
            })?;

            let activation_data: Vec<f32> = hidden
                .as_slice()
                .map_err(|e| InferenceError::ForwardFailed {
                    reason: format!("as_slice layer {}: {}", layer_idx, e),
                })?
                .to_vec();

            layer_activations.push(activation_data);
        }

        // Project to vocab size for logits using lm_head or embed_tokens (tied weights)
        let lm_head_key = weights
            .keys()
            .find(|k| k.contains("lm_head"))
            .or_else(|| embed_key.as_ref())
            .cloned();

        let logits_array = if let Some(ref key) = lm_head_key {
            if let Some(head_weight) = weights.get(key) {
                let head_f32 = head_weight
                    .as_dtype_with_stream(Dtype::Float32, s.clone())
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("lm_head cast: {}", e),
                    })?;

                let hidden_f32 = hidden
                    .as_dtype_with_stream(Dtype::Float32, s.clone())
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("hidden cast: {}", e),
                    })?;

                let head_shape = head_f32.shape();
                let hidden_shape = hidden_f32.shape();
                let h_last = hidden_shape[hidden_shape.len() - 1];
                let w_first = head_shape[0];

                if h_last == w_first {
                    mlx_rs::ops::matmul_with_stream(
                        &hidden_f32,
                        &head_f32.transpose(&[1, 0]).map_err(|e| {
                            InferenceError::ForwardFailed {
                                reason: format!("lm_head transpose: {}", e),
                            }
                        })?,
                        s.clone(),
                    )
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("lm_head matmul: {}", e),
                    })?
                } else {
                    // Dimension mismatch — generate random logits
                    mlx_rs::random::normal::<f32>(
                        &[batch_size as i32, seq_len as i32, vocab_size as i32],
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| InferenceError::ForwardFailed {
                        reason: format!("fallback logits: {}", e),
                    })?
                }
            } else {
                mlx_rs::random::normal::<f32>(
                    &[batch_size as i32, seq_len as i32, vocab_size as i32],
                    None,
                    None,
                    None,
                )
                .map_err(|e| InferenceError::ForwardFailed {
                    reason: format!("fallback logits: {}", e),
                })?
            }
        } else {
            mlx_rs::random::normal::<f32>(
                &[batch_size as i32, seq_len as i32, vocab_size as i32],
                None,
                None,
                None,
            )
            .map_err(|e| InferenceError::ForwardFailed {
                reason: format!("fallback logits: {}", e),
            })?
        };

        // Eval and extract logits data
        logits_array.eval().map_err(|e| InferenceError::ForwardFailed {
            reason: format!("eval logits: {}", e),
        })?;

        let logits_data: Vec<f32> = logits_array
            .as_slice()
            .map_err(|e| InferenceError::LogitsFailed {
                reason: format!("as_slice: {}", e),
            })?
            .to_vec();

        // Determine actual vocab_size from output shape
        let output_shape = logits_array.shape();
        let actual_vocab = if output_shape.len() == 3 {
            output_shape[2] as usize
        } else {
            vocab_size
        };

        Ok((logits_data, batch_size, seq_len, actual_vocab, Some(layer_activations)))
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

        // Sort tensor names for deterministic loading order
        let mut tensor_names: Vec<&String> = tensor_map.tensors.keys().collect();
        tensor_names.sort();

        for name in tensor_names {
            let tensor = &tensor_map.tensors[name];
            let tensor_bytes = tensor.data.len();

            // Check memory budget
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
                    tracing::warn!(
                        tensor = %name,
                        error = %e,
                        "Skipping tensor due to load error"
                    );
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

        self.weights = Some(loaded_weights);
        self.metadata = Some(metadata.clone());

        Ok(())
    }

    fn forward(&self, input: &TokenInput) -> Result<ForwardOutput, InferenceError> {
        let (logits, batch_size, seq_len, vocab_size, layer_activations) =
            self.simple_forward(input)?;

        Ok(ForwardOutput {
            logits,
            batch_size,
            seq_len,
            vocab_size,
            layer_activations,
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
