//! ADR-037 Phase E4b — EAGLE-3 drafter forward primitives.
//!
//! Mirrors `src/inference/spec_decode/dflash/forward.rs` pattern but
//! adapted for the EAGLE-3 1-layer drafter. Each function dispatches
//! one stage of the forward pass; the orchestrator (Phase E4b.6,
//! future iter) chains them.
//!
//! ## Current sub-phase coverage
//!
//! - **E4b.1** `tensors.rs`: GPU upload pipeline (SHIPPED).
//! - **E4b.2** `dispatch_eagle3_fc` (this file): projects
//!   `[seq, num_aux * H]` → `[seq, H]` via the BF16 `fc.weight`.
//! - **E4b.3+** TODO: input_layernorm, hidden_norm, concat,
//!   self-attn (via Phase E1 `tree_attention` kernel), MLP, final
//!   norm, lm_head, top-K extraction.

use super::config::Eagle3DrafterConfig;
use super::tensors::Eagle3DrafterTensors;
use crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;
use anyhow::{anyhow, Context, Result};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

/// EAGLE-3 FC projection.
///
/// Mirrors vLLM `Eagle3LlamaForCausalLM::combine_hidden_states` last
/// step (`llama_eagle3.py:407` `return self.model.fc(hidden_states)`).
/// Projects the concatenated multi-aux hidden tensor
/// `[seq, num_aux * target_hidden_size]` (which `Eagle3HiddenCollector`
/// produces) down to the drafter's hidden_size:
///
/// ```text
///   output[s, h] = sum over k of input[s, k] * fc_weight[h, k]
/// ```
///
/// Where `fc_weight: [hidden_size, fc_input_size]` BF16 and
/// `fc_input_size = num_aux * target_hidden_size`.
///
/// # Arguments
/// * `concat_hidden_gpu` — F32 buffer of shape
///   `[seq_len, fc_input_size]`. Caller uploads `Eagle3HiddenCollector::
///   concatenated_hidden()` to GPU before this call.
/// * `tensors` — uploaded drafter weights from Phase E4b.1.
/// * `cfg` — drafter config (provides expected dims for validation).
/// * `seq_len` — number of token positions in `concat_hidden_gpu`.
///
/// Returns the output buffer `[seq_len, hidden_size]` F32. Caller
/// commits the encoder.
pub fn dispatch_eagle3_fc(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    concat_hidden_gpu: &MlxBuffer,
    tensors: &Eagle3DrafterTensors,
    cfg: &Eagle3DrafterConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    // Codex /cfa E4b.2 Critical (2026-05-22): input dtype must be
    // F32. apply_linear_projection_f32 only debug-asserts this;
    // a release build with BF16/U8 input would mis-stride reads.
    // Fail-fast here at the wrapper.
    if concat_hidden_gpu.dtype() != DType::F32 {
        return Err(anyhow!(
            "dispatch_eagle3_fc: concat_hidden dtype must be F32, got {:?}",
            concat_hidden_gpu.dtype()
        ));
    }
    // Codex /cfa E4b.2 Major (2026-05-22): use try_from instead of
    // `as u32` to catch silent truncation on adversarial config.
    let fc_in_usize = cfg.fc_input_size();
    let hidden_usize = cfg.hidden_size;
    let fc_in: u32 = u32::try_from(fc_in_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_fc: fc_input_size ({}) exceeds u32::MAX",
            fc_in_usize
        )
    })?;
    let hidden: u32 = u32::try_from(hidden_usize).map_err(|_| {
        anyhow!(
            "dispatch_eagle3_fc: hidden_size ({}) exceeds u32::MAX",
            hidden_usize
        )
    })?;
    // Codex /cfa E4b.2 Major (2026-05-22): checked multiply for the
    // expected element count. Otherwise large seq_len could wrap on
    // release and let an undersized buffer pass validation.
    let expected_input_elems = (seq_len as usize)
        .checked_mul(fc_in_usize)
        .ok_or_else(|| {
            anyhow!(
                "dispatch_eagle3_fc: seq_len ({}) * fc_input_size ({}) overflows usize",
                seq_len,
                fc_in_usize
            )
        })?;
    let actual_elems = concat_hidden_gpu.element_count();
    if actual_elems != expected_input_elems {
        return Err(anyhow!(
            "dispatch_eagle3_fc: concat_hidden has {} elements, expected {} (seq_len={} * fc_input_size={})",
            actual_elems, expected_input_elems, seq_len, fc_in
        ));
    }
    apply_linear_projection_f32(
        encoder, registry, device,
        concat_hidden_gpu, &tensors.fc,
        seq_len, fc_in, hidden,
    )
    .context("dispatch_eagle3_fc")
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::eagle3::weights::{
        expected_manifest, Eagle3Weights, ExpectedTensor,
    };
    use mlx_native::DType;
    use safetensors::tensor::{Dtype as SafeDtype, TensorView};
    use std::collections::BTreeMap;

    /// Small config so synthetic-blob tests run fast.
    fn tiny_cfg() -> Eagle3DrafterConfig {
        Eagle3DrafterConfig {
            hidden_size: 256,
            intermediate_size: 512,
            head_dim: 32,
            num_q_heads: 8,
            num_kv_heads: 4,
            vocab_size: 1000,
            draft_vocab_size: 1000,
            target_hidden_size: 256,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: false, // simpler manifest for parity test
            use_qk_norm: false,
            attention_bias: false,
            tie_lm_head: true, // simpler
            include_draft_id_mapping: false,
            has_own_embed_tokens: false,
        }
    }

    /// Truncate F32 → BF16 (top 16 bits) and serialize as little-endian.
    fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            let bf16_bits = (v.to_bits() >> 16) as u16;
            out.push((bf16_bits & 0xff) as u8);
            out.push(((bf16_bits >> 8) & 0xff) as u8);
        }
        out
    }

    /// Round F32 to its BF16-representable value (truncate low 16 bits
    /// of the F32 bit pattern, then promote back to F32). Used in the
    /// CPU reference so the parity comparison accounts for the BF16
    /// quantization of the on-disk weight.
    fn bf16_quantize_f32(v: f32) -> f32 {
        let bits = v.to_bits() & 0xFFFF0000;
        f32::from_bits(bits)
    }

    /// CPU reference matmul: output[s, h] = sum_k input[s, k] * weight[h, k].
    /// Uses f64 accumulator for precision.
    fn cpu_fc_reference(
        input: &[f32],         // [seq, in_features]
        weight_bf16_q: &[f32], // [out_features, in_features], pre-BF16-quantized
        seq_len: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * out_features];
        for s in 0..seq_len {
            for h in 0..out_features {
                let mut acc = 0.0f64;
                for k in 0..in_features {
                    acc += (input[s * in_features + k] as f64)
                        * (weight_bf16_q[h * in_features + k] as f64);
                }
                out[s * out_features + h] = acc as f32;
            }
        }
        out
    }

    /// Deterministic pseudo-random F32 in [-1, 1).
    fn pseudo_random(seed: u64) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((x >> 33) as u32) & 0x7FFFFF;
        (bits as f32 / 0x7FFFFF as f32) * 2.0 - 1.0
    }

    fn fill_random(buf: &mut [f32], seed: u64) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = pseudo_random(seed.wrapping_add(i as u64));
        }
    }

    /// Build a safetensors blob with custom `fc.weight` bytes. All
    /// other manifest tensors get zero bytes (sufficient for upload —
    /// we only consume `fc` in this test).
    fn build_blob_with_fc_weight(
        manifest: &[ExpectedTensor],
        fc_bytes: &[u8],
    ) -> Vec<u8> {
        let mut storage: Vec<Vec<u8>> = Vec::with_capacity(manifest.len());
        for exp in manifest {
            let elem_bytes = match exp.dtype {
                SafeDtype::BF16 => 2,
                SafeDtype::I64 => 8,
                _ => panic!("unexpected dtype in test"),
            };
            let nelem: usize = exp.shape.iter().product();
            if exp.name == "fc.weight" {
                assert_eq!(fc_bytes.len(), nelem * elem_bytes);
                storage.push(fc_bytes.to_vec());
            } else {
                storage.push(vec![0u8; nelem * elem_bytes]);
            }
        }
        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        for (i, exp) in manifest.iter().enumerate() {
            let view =
                TensorView::new(exp.dtype, exp.shape.clone(), storage[i].as_slice())
                    .expect("synthetic view");
            tensors.insert(exp.name.clone(), view);
        }
        safetensors::serialize(
            &tensors,
            None::<std::collections::HashMap<String, String>>,
        )
        .expect("serialize")
    }

    fn upload_f32_to_gpu(
        device: &MlxDevice,
        data: &[f32],
        shape: Vec<usize>,
    ) -> MlxBuffer {
        let bytes = data.len() * 4;
        let mut buf = device
            .alloc_buffer(bytes, DType::F32, shape)
            .expect("alloc input");
        buf.as_mut_slice::<f32>()
            .expect("input slice")
            .copy_from_slice(data);
        buf
    }

    #[test]
    fn adr_037_e4b2_fc_cpu_parity_seq_4_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);

        let seq_len: u32 = 4;
        let fc_in = cfg.fc_input_size();
        let hidden = cfg.hidden_size;

        // Synthesize input (F32) and weight (F32 reference + BF16 truncation
        // for both the GPU buffer and the CPU reference matmul).
        let mut input_data = vec![0.0f32; (seq_len as usize) * fc_in];
        fill_random(&mut input_data, 0xA00);

        let mut weight_f32 = vec![0.0f32; hidden * fc_in];
        fill_random(&mut weight_f32, 0xB00);
        let weight_bf16_bytes = f32_to_bf16_bytes(&weight_f32);
        // BF16-quantized F32 (what the GPU kernel effectively uses).
        let weight_bf16_q: Vec<f32> = weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();

        // CPU reference using BF16-quantized weights → matches what GPU computes.
        let cpu_out =
            cpu_fc_reference(&input_data, &weight_bf16_q, seq_len as usize, fc_in, hidden);

        // GPU path: build synthetic safetensors blob, load + upload, dispatch.
        let blob = build_blob_with_fc_weight(&manifest, &weight_bf16_bytes);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights)
            .expect("upload tensors");
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, fc_in],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf =
            dispatch_eagle3_fc(&mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len)
                .expect("dispatch_eagle3_fc");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(
            gpu_out.len(),
            (seq_len as usize) * hidden,
            "output shape"
        );

        // Compare GPU vs CPU within tolerance. BF16 quantized weights ×
        // F32 input × seq=4 accumulation gives relative error ~1e-3
        // on random inputs; absolute error scales with input/weight
        // magnitudes (both in [-1,1)) and the inner-dim accumulation
        // size (768). Tolerance 5e-2 is conservative.
        let mut max_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 5e-2,
                "FC parity violated at idx {i}: gpu={g} cpu={c} diff={d}"
            );
        }
        eprintln!("fc parity seq=4 max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b2_fc_cpu_parity_seq_1_decode_2026_05_22() {
        // seq=1 exercises the GEMV path (dense_gemv_bf16_f32) instead
        // of the tiled GEMM path.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);

        let _seq_len: u32 = 1; // documents intent; cpu_fc_reference uses literal 1 below
        let fc_in = cfg.fc_input_size();
        let hidden = cfg.hidden_size;

        let mut input_data = vec![0.0f32; fc_in];
        fill_random(&mut input_data, 0xC00);
        let mut weight_f32 = vec![0.0f32; hidden * fc_in];
        fill_random(&mut weight_f32, 0xD00);
        let weight_bf16_bytes = f32_to_bf16_bytes(&weight_f32);
        let weight_bf16_q: Vec<f32> = weight_f32.iter().map(|&v| bf16_quantize_f32(v)).collect();

        let cpu_out = cpu_fc_reference(&input_data, &weight_bf16_q, 1, fc_in, hidden);

        let blob = build_blob_with_fc_weight(&manifest, &weight_bf16_bytes);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let input_gpu = upload_f32_to_gpu(&device, &input_data, vec![1, fc_in]);

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf =
            dispatch_eagle3_fc(&mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, 1)
                .expect("dispatch_eagle3_fc (seq=1)");
        enc.commit_and_wait().expect("commit");

        let gpu_out: &[f32] = out_buf.as_slice::<f32>().expect("output slice");
        assert_eq!(gpu_out.len(), hidden);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < 5e-2, "seq=1 GEMV parity: diff={d} > 5e-2");
        }
        eprintln!("fc parity seq=1 (GEMV) max_diff={max_diff:.6e}");
    }

    #[test]
    fn adr_037_e4b2_fc_rejects_wrong_input_shape_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_fc_weight(
            &manifest,
            &vec![0u8; cfg.hidden_size * cfg.fc_input_size() * 2],
        );
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        // Allocate input with wrong size — should be seq_len * fc_in
        // = 4 * 768 = 3072 floats. Allocate 100 instead.
        let bad_data = vec![0.0f32; 100];
        let bad_input = upload_f32_to_gpu(&device, &bad_data, vec![100]);

        let mut enc = device.command_encoder().expect("encoder");
        let err =
            dispatch_eagle3_fc(&mut enc, &mut registry, &device, &bad_input, &tensors, &cfg, 4)
                .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("concat_hidden has"),
            "expected shape error, got: {msg}"
        );
    }

    #[test]
    fn adr_037_e4b2_gate_fc_rejects_non_f32_input_dtype_2026_05_22() {
        // Codex /cfa E4b.2 Critical fix (2026-05-22): wrapper must
        // reject non-F32 input even though apply_linear_projection_f32
        // only debug-asserts this. Validates the hard check.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_fc_weight(
            &manifest,
            &vec![0u8; cfg.hidden_size * cfg.fc_input_size() * 2],
        );
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        // Allocate a BF16 input with the correct element count. Wrapper
        // should reject due to wrong dtype, not pass through.
        let seq_len = 2_u32;
        let elem_count = (seq_len as usize) * cfg.fc_input_size();
        let bad_input = device
            .alloc_buffer(
                elem_count * 2, // BF16 size
                DType::BF16,
                vec![seq_len as usize, cfg.fc_input_size()],
            )
            .expect("alloc bad input");

        let mut enc = device.command_encoder().expect("encoder");
        let err = dispatch_eagle3_fc(
            &mut enc, &mut registry, &device, &bad_input, &tensors, &cfg, seq_len,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("dtype must be F32"),
            "expected F32-dtype error, got: {msg}"
        );
    }

    #[test]
    fn adr_037_e4b2_fc_output_shape_correct_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_blob_with_fc_weight(
            &manifest,
            &vec![0u8; cfg.hidden_size * cfg.fc_input_size() * 2],
        );
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        let seq_len = 8_u32;
        let input_data = vec![0.0f32; (seq_len as usize) * cfg.fc_input_size()];
        let input_gpu = upload_f32_to_gpu(
            &device,
            &input_data,
            vec![seq_len as usize, cfg.fc_input_size()],
        );

        let mut enc = device.command_encoder().expect("encoder");
        let out_buf = dispatch_eagle3_fc(
            &mut enc, &mut registry, &device, &input_gpu, &tensors, &cfg, seq_len,
        )
        .expect("dispatch");
        enc.commit_and_wait().expect("commit");
        assert_eq!(out_buf.dtype(), DType::F32);
        assert_eq!(
            out_buf.element_count(),
            (seq_len as usize) * cfg.hidden_size
        );
    }
}
