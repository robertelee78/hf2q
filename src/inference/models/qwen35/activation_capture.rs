//! Activation capture interface for DWQ calibration (ADR-013 Decision 16).
//!
//! Defines the cross-ADR contract between ADR-013 (inference) and ADR-012
//! (DWQ conversion). ADR-012's P6 runs a calibration pass to collect
//! per-layer activation statistics for activation-aware quantization; the
//! `ActivationCapture` trait is the runtime-agnostic interface it consumes.
//!
//! # Why a trait, not a concrete method
//!
//! ADR-012 needs to run calibration while ADR-013's end-to-end forward
//! pass is still in flight. The trait lets ADR-012 wire up against a
//! `Box<dyn ActivationCapture>` today; hf2q's real implementation on
//! `Qwen35Model` lands after P11 (end-to-end wire-up) without changing
//! the ADR-012 side.
//!
//! # Mock implementation
//!
//! [`MockActivationCapture`] returns deterministic synthetic activations
//! that match the `LayerActivations` shape. It's intended for:
//!
//! 1. ADR-012's calibration tests (they can inject the mock instead of
//!    a real model to validate their accumulator logic).
//! 2. hf2q-side trait-impl smoke tests (this file).
//!
//! # Acceptance (ADR-013 Decision 16)
//!
//! * Trait defined in hf2q before ADR-012 P6 starts wiring. ✓
//! * Mock impl exists for ADR-012-side tests. ✓ (MockActivationCapture)
//! * Real impl on Qwen35Model — **P11 follow-up** (requires working forward).
//! * No dependency on end-to-end inference at merge time. ✓

use anyhow::Result;

// ================================================================
// Data types
// ================================================================

/// Per-layer activation snapshots collected during a calibration forward pass.
///
/// Layout: `layer_inputs[layer_idx]` is the residual stream entering layer
/// `layer_idx` (after `token_embd` for layer 0, after the previous layer's
/// residual add for subsequent layers). `layer_outputs[layer_idx]` is the
/// residual stream leaving the same layer.
///
/// Each entry is flattened f32 row-major `[seq_len, hidden_size]`.
///
/// # Why both inputs and outputs
///
/// ADR-012's DWQ calibration needs **inputs** (to compute activation-range
/// statistics for quantizer calibration — "what values does this layer see?")
/// and sometimes **outputs** (to verify quantization error propagation).
/// Most quantizers use inputs only; outputs are recorded for completeness.
#[derive(Debug, Clone)]
pub struct LayerActivations {
    /// Per-layer residual-stream inputs. `len() == num_layers`.
    pub layer_inputs: Vec<Vec<f32>>,
    /// Per-layer residual-stream outputs. `len() == num_layers`.
    pub layer_outputs: Vec<Vec<f32>>,
    /// Number of hidden layers captured (== `len()` of both vecs above).
    pub num_layers: u32,
    /// Sequence length of the calibration prompt.
    pub seq_len: u32,
    /// Residual-stream hidden dimension.
    pub hidden_size: u32,
}

impl LayerActivations {
    /// Total number of f32 values across all captured tensors (for memory
    /// accounting / logs).
    pub fn element_count(&self) -> usize {
        (self.num_layers as usize) * (self.seq_len as usize) * (self.hidden_size as usize) * 2
    }

    /// Validate internal shape consistency. Returns the first error, or Ok.
    pub fn validate(&self) -> Result<()> {
        if self.layer_inputs.len() != self.num_layers as usize {
            anyhow::bail!(
                "LayerActivations: layer_inputs.len() = {} != num_layers = {}",
                self.layer_inputs.len(),
                self.num_layers
            );
        }
        if self.layer_outputs.len() != self.num_layers as usize {
            anyhow::bail!(
                "LayerActivations: layer_outputs.len() = {} != num_layers = {}",
                self.layer_outputs.len(),
                self.num_layers
            );
        }
        let expected_per_layer = (self.seq_len as usize) * (self.hidden_size as usize);
        for (i, v) in self.layer_inputs.iter().enumerate() {
            if v.len() != expected_per_layer {
                anyhow::bail!(
                    "LayerActivations: layer_inputs[{}].len() = {} != seq_len({}) * hidden_size({}) = {}",
                    i,
                    v.len(),
                    self.seq_len,
                    self.hidden_size,
                    expected_per_layer
                );
            }
        }
        for (i, v) in self.layer_outputs.iter().enumerate() {
            if v.len() != expected_per_layer {
                anyhow::bail!(
                    "LayerActivations: layer_outputs[{}].len() = {} != {}",
                    i,
                    v.len(),
                    expected_per_layer
                );
            }
        }
        Ok(())
    }
}

// ================================================================
// Trait
// ================================================================

/// Runtime-agnostic calibration interface (ADR-013 Decision 16).
///
/// ADR-012's DWQ calibration pass consumes this. Implementations run a
/// forward pass through a full model (or any calibration harness) and
/// return per-layer activation snapshots.
///
/// # Lifecycle
///
/// * `run_calibration_prompt(tokens)` may be called multiple times with
///   different prompts. Each call produces a fresh `LayerActivations`;
///   the caller aggregates across calls if their quantizer needs a
///   multi-prompt average.
///
/// * Implementations are free to mutate internal state (e.g. zero KV
///   cache between calls), hence `&mut self`.
///
/// # Errors
///
/// Any failure in the forward pass (tokenizer, kernel, OOM, etc.) returns
/// `Err`. Implementations should not panic.
pub trait ActivationCapture {
    /// Run a forward pass over `tokens` and capture per-layer activation
    /// snapshots.
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations>;
}

// ================================================================
// Mock implementation
// ================================================================

/// Deterministic synthetic [`ActivationCapture`] for tests.
///
/// Returns activations shaped by the configured `num_layers`, `seq_len`
/// (derived from `tokens.len()`), and `hidden_size`. Contents are
/// deterministic as a function of `tokens`, layer index, and position —
/// no randomness, no state — so ADR-012 tests can pin expected values
/// in a round-trip.
pub struct MockActivationCapture {
    pub num_layers: u32,
    pub hidden_size: u32,
}

impl MockActivationCapture {
    pub fn new(num_layers: u32, hidden_size: u32) -> Self {
        Self {
            num_layers,
            hidden_size,
        }
    }
}

impl ActivationCapture for MockActivationCapture {
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations> {
        if tokens.is_empty() {
            anyhow::bail!("MockActivationCapture: tokens must be non-empty");
        }
        let seq = tokens.len();
        let h = self.hidden_size as usize;
        let mut layer_inputs = Vec::with_capacity(self.num_layers as usize);
        let mut layer_outputs = Vec::with_capacity(self.num_layers as usize);

        // Deterministic function: input[l][t, j] = (token[t] * 0.001 + l * 0.01 + j * 0.0001).
        // Output differs by + 1.0 so tests can distinguish.
        for l in 0..self.num_layers as usize {
            let mut inp = vec![0.0f32; seq * h];
            let mut out = vec![0.0f32; seq * h];
            for t in 0..seq {
                for j in 0..h {
                    let base = (tokens[t] as f32) * 0.001
                        + (l as f32) * 0.01
                        + (j as f32) * 0.0001;
                    inp[t * h + j] = base;
                    out[t * h + j] = base + 1.0;
                }
            }
            layer_inputs.push(inp);
            layer_outputs.push(out);
        }

        Ok(LayerActivations {
            layer_inputs,
            layer_outputs,
            num_layers: self.num_layers,
            seq_len: seq as u32,
            hidden_size: self.hidden_size,
        })
    }
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_validate_accepts_matching_shapes() {
        let act = LayerActivations {
            layer_inputs: vec![vec![0.0; 8]; 3],
            layer_outputs: vec![vec![0.0; 8]; 3],
            num_layers: 3,
            seq_len: 2,
            hidden_size: 4,
        };
        act.validate().expect("should validate");
    }

    #[test]
    fn activations_validate_rejects_wrong_layer_count() {
        let act = LayerActivations {
            layer_inputs: vec![vec![0.0; 8]; 2], // only 2 layers
            layer_outputs: vec![vec![0.0; 8]; 3],
            num_layers: 3,
            seq_len: 2,
            hidden_size: 4,
        };
        assert!(act.validate().is_err());
    }

    #[test]
    fn activations_validate_rejects_wrong_element_count() {
        let act = LayerActivations {
            layer_inputs: vec![vec![0.0; 9]; 3], // wrong per-layer element count
            layer_outputs: vec![vec![0.0; 8]; 3],
            num_layers: 3,
            seq_len: 2,
            hidden_size: 4,
        };
        assert!(act.validate().is_err());
    }

    #[test]
    fn activations_element_count_matches_expected() {
        let act = LayerActivations {
            layer_inputs: vec![vec![0.0; 8]; 3],
            layer_outputs: vec![vec![0.0; 8]; 3],
            num_layers: 3,
            seq_len: 2,
            hidden_size: 4,
        };
        assert_eq!(act.element_count(), 3 * 2 * 4 * 2);
    }

    #[test]
    fn mock_produces_correct_shapes() {
        let mut mock = MockActivationCapture::new(5, 16);
        let tokens = vec![10u32, 20, 30];
        let act = mock.run_calibration_prompt(&tokens).expect("run");
        act.validate().expect("shapes valid");

        assert_eq!(act.num_layers, 5);
        assert_eq!(act.seq_len, 3);
        assert_eq!(act.hidden_size, 16);
        assert_eq!(act.layer_inputs.len(), 5);
        assert_eq!(act.layer_outputs.len(), 5);
        for v in &act.layer_inputs {
            assert_eq!(v.len(), 3 * 16);
        }
        for v in &act.layer_outputs {
            assert_eq!(v.len(), 3 * 16);
        }
    }

    #[test]
    fn mock_is_deterministic() {
        let mut mock1 = MockActivationCapture::new(2, 4);
        let mut mock2 = MockActivationCapture::new(2, 4);
        let tokens = vec![1u32, 2, 3];
        let a1 = mock1.run_calibration_prompt(&tokens).unwrap();
        let a2 = mock2.run_calibration_prompt(&tokens).unwrap();
        for l in 0..2 {
            for i in 0..12 {
                assert_eq!(
                    a1.layer_inputs[l][i].to_bits(),
                    a2.layer_inputs[l][i].to_bits()
                );
                assert_eq!(
                    a1.layer_outputs[l][i].to_bits(),
                    a2.layer_outputs[l][i].to_bits()
                );
            }
        }
    }

    #[test]
    fn mock_differs_between_layers() {
        let mut mock = MockActivationCapture::new(3, 4);
        let tokens = vec![5u32];
        let act = mock.run_calibration_prompt(&tokens).unwrap();
        let l0 = &act.layer_inputs[0];
        let l1 = &act.layer_inputs[1];
        let l2 = &act.layer_inputs[2];
        // Mock formula increments by 0.01 per layer → adjacent layers differ.
        let mut any_differ_01 = false;
        let mut any_differ_12 = false;
        for i in 0..4 {
            if (l0[i] - l1[i]).abs() > 1e-6 {
                any_differ_01 = true;
            }
            if (l1[i] - l2[i]).abs() > 1e-6 {
                any_differ_12 = true;
            }
        }
        assert!(any_differ_01, "layer 0 == layer 1 (mock broken)");
        assert!(any_differ_12, "layer 1 == layer 2 (mock broken)");
    }

    #[test]
    fn mock_differs_between_tokens() {
        let mut mock = MockActivationCapture::new(1, 4);
        let a1 = mock
            .run_calibration_prompt(&[1u32, 2, 3])
            .unwrap();
        let a2 = mock
            .run_calibration_prompt(&[100u32, 200, 300])
            .unwrap();
        // Different tokens → different inputs.
        let mut any_differ = false;
        for i in 0..12 {
            if (a1.layer_inputs[0][i] - a2.layer_inputs[0][i]).abs() > 1e-6 {
                any_differ = true;
                break;
            }
        }
        assert!(any_differ, "mock token content is not encoded in activations");
    }

    #[test]
    fn mock_rejects_empty_tokens() {
        let mut mock = MockActivationCapture::new(2, 4);
        let result = mock.run_calibration_prompt(&[]);
        assert!(result.is_err(), "empty tokens should error");
    }

    /// Demonstrates the cross-ADR usage pattern: ADR-012's calibration
    /// accumulator can consume any `ActivationCapture` impl via trait
    /// object without caring whether it's a real model or a mock.
    #[test]
    fn trait_object_usable_for_cross_adr_consumption() {
        fn accumulate_mean_input_per_layer(
            capture: &mut dyn ActivationCapture,
            tokens: &[u32],
        ) -> Result<Vec<f32>> {
            let act = capture.run_calibration_prompt(tokens)?;
            let mut means = Vec::with_capacity(act.num_layers as usize);
            for l in 0..act.num_layers as usize {
                let sum: f32 = act.layer_inputs[l].iter().sum();
                let n = act.layer_inputs[l].len() as f32;
                means.push(sum / n);
            }
            Ok(means)
        }

        let mut mock = MockActivationCapture::new(4, 8);
        let tokens = vec![42u32, 100, 7];
        let means = accumulate_mean_input_per_layer(&mut mock, &tokens).expect("accum");
        assert_eq!(means.len(), 4);
        // Means should be finite and strictly monotonic in layer index
        // because mock formula adds 0.01 per layer.
        for i in 1..means.len() {
            assert!(means[i] > means[i - 1], "layer {} mean should exceed layer {}", i, i - 1);
        }
    }
}
