//! Weight-buffer payload accounting, in the allocation order of `upload`.
//!
//! The checkpoint is a read-only, pageable mmap, not a second heap copy and not
//! pinned GPU storage. These byte counts exclude that mapping, allocator/page
//! rounding, Rust object headers and the borrowed target GGUF head. They are not
//! a process-RSS or whole-device-residency estimate.

use super::{DFlash2Config, DFlash2ModelTensors, DFlash2Projection, DFlash2TensorsError, ProjectionFormat};
use mlx_native::MlxBuffer;

type Result<T, E = DFlash2TensorsError> = std::result::Result<T, E>;

fn invalid(message: &str) -> DFlash2TensorsError {
    DFlash2TensorsError::Memory(message.into())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| invalid("byte/count addition overflow"))
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("byte/count multiplication overflow"))
}

fn allocation(bytes: usize) -> Result<usize> {
    if bytes == 0 || bytes > isize::MAX as usize {
        return Err(invalid("allocation must be nonzero and fit isize"));
    }
    Ok(bytes)
}

pub(super) struct ProjectionMemory {
    pub retained_bytes: usize,
    pub groups: usize,
    pub code_words: usize,
    temporary_bytes: usize,
    roots: usize,
}

impl ProjectionMemory {
    pub(super) fn new(rows: usize, columns: usize, format: ProjectionFormat) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(invalid("projection dimensions must be nonzero"));
        }
        let elements = mul(rows, columns)?;
        // Source tensors and BF16 fusion must also be representable, even for AF4.
        let bf16_bytes = allocation(mul(elements, 2)?)?;
        match format {
            ProjectionFormat::Bf16 => Ok(Self {
                retained_bytes: bf16_bytes,
                groups: 0,
                code_words: 0,
                temporary_bytes: bf16_bytes,
                roots: 1,
            }),
            ProjectionFormat::Af4G64 => {
                if rows % 64 != 0 || columns % 64 != 0 {
                    return Err(invalid("AF4G64 requires 64-aligned rows and columns"));
                }
                let groups = elements / 64;
                let code_words = mul(groups, 8)?;
                let codes = allocation(mul(code_words, 4)?)?;
                let metadata = allocation(mul(groups, 4)?)?;
                let retained_bytes = add(codes, metadata)?;
                Ok(Self {
                    retained_bytes,
                    groups,
                    code_words,
                    // Row code_words + scale/bias Vecs coexist with the tiled
                    // prepared_words + metadata Vecs and both GPU roots.
                    // append_af4 uses only a 64-f32 stack group, never a full
                    // F32 matrix; that earlier phase is below this high-water.
                    temporary_bytes: mul(retained_bytes, 2)?,
                    roots: 2,
                })
            }
        }
    }
}

/// Requested buffer/Vec payload bytes, not allocator-rounded resident bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DFlash2WeightMemory {
    pub retained_bytes: usize,
    pub retained_buffer_count: usize,
    /// Maximum of already-uploaded roots plus the current upload's host Vecs.
    /// This is not `retained_bytes + upload_temporary_peak_bytes`: those maxima
    /// need not occur at the same point in the serial load order.
    pub upload_peak_bytes: usize,
    pub upload_temporary_peak_bytes: usize,
    /// Global uploads precede allocation of the layer Vec backing.
    pub global_upload_peak_bytes: usize,
    /// Includes global retained roots and all preceding complete layers.
    pub layer_upload_peak_bytes: usize,
}

impl DFlash2WeightMemory {
    fn root(&mut self, bytes: usize) -> Result<()> {
        self.upload(bytes, 0, 1)
    }

    fn upload(&mut self, bytes: usize, temporary: usize, roots: usize) -> Result<()> {
        self.retained_bytes = add(self.retained_bytes, bytes)?;
        self.retained_buffer_count = add(self.retained_buffer_count, roots)?;
        self.upload_peak_bytes = self.upload_peak_bytes.max(add(self.retained_bytes, temporary)?);
        self.upload_temporary_peak_bytes = self.upload_temporary_peak_bytes.max(temporary);
        Ok(())
    }

    fn projection(&mut self, rows: usize, columns: usize, format: ProjectionFormat) -> Result<()> {
        let memory = ProjectionMemory::new(rows, columns, format)?;
        self.upload(memory.retained_bytes, memory.temporary_bytes, memory.roots)
    }
}

impl DFlash2ModelTensors {
    /// Preallocation sizing for a validated checkpoint descriptor and an explicit
    /// selected representation. `has_mask_embedding` comes from the validated
    /// schema, not checkpoint file size. No environment is read here.
    pub fn weight_memory(
        config: &DFlash2Config,
        format: ProjectionFormat,
        has_mask_embedding: bool,
    ) -> Result<DFlash2WeightMemory> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        if hidden == 0 || intermediate == 0 || config.num_hidden_layers == 0
            || config.num_attention_heads == 0 || config.num_key_value_heads == 0
            || config.head_dim == 0 || config.vocab_size == 0 || config.selector_rank == 0
            || config.target_layer_ids.is_empty() || config.conv_group_size == 0
            || hidden % config.conv_group_size != 0 || config.conv_kernel_size != 2
            || config.num_attention_heads % config.num_key_value_heads != 0
        {
            return Err(invalid("unsupported or empty validated checkpoint geometry"));
        }
        let query = mul(config.num_attention_heads, config.head_dim)?;
        let kv = mul(config.num_key_value_heads, config.head_dim)?;
        let context_kv = mul(2, kv)?;
        let qkv = add(query, context_kv)?;
        if format == ProjectionFormat::Af4G64 && (query % 64 != 0 || context_kv % 64 != 0) {
            return Err(invalid("AF4G64 context-KV view must start and end on 64-row tiles"));
        }
        let gate_up = mul(2, intermediate)?;
        let fc_input = mul(hidden, config.target_layer_ids.len())?;
        let dynamic = mul(mul(2, config.conv_kernel_size)?, hidden / config.conv_group_size)?;
        let norm = allocation(mul(hidden, 4)?)?;
        let head_norm = allocation(mul(config.head_dim, 4)?)?;
        let convolution = allocation(mul(mul(2, config.conv_kernel_size)?, mul(hidden, 2)?)?)?;
        let codebook = allocation(mul(mul(config.vocab_size, config.selector_rank)?, 2)?)?;

        let mut global = DFlash2WeightMemory::default();
        global.projection(hidden, fc_input, format)?;
        global.root(norm)?;
        global.root(norm)?;
        global.projection(config.selector_rank, hidden, format)?;
        global.root(codebook)?;
        global.root(codebook)?;
        if has_mask_embedding {
            global.root(allocation(mul(hidden, 2)?)?)?;
        }

        let mut layer = DFlash2WeightMemory::default();
        // These locals upload before the struct initializer uploads its fields.
        layer.projection(qkv, hidden, format)?;
        // context_kv_projection and gate_up_halves are views of their roots.
        layer.projection(gate_up, hidden, format)?;
        layer.projection(dynamic, hidden, format)?;
        layer.root(convolution)?;
        layer.root(norm)?;
        layer.projection(hidden, query, format)?;
        layer.root(head_norm)?;
        layer.root(head_norm)?;
        layer.projection(dynamic, hidden, format)?;
        layer.root(norm)?;
        layer.projection(hidden, intermediate, format)?;
        layer.root(convolution)?;

        // Identical layer geometry: only the last layer can set the layer-phase
        // high-water. Avoid iterating an untrusted number of layers to size it.
        let preceding_layers = mul(config.num_hidden_layers - 1, layer.retained_bytes)?;
        let layer_peak = add(add(global.retained_bytes, preceding_layers)?, layer.upload_peak_bytes)?;
        Ok(DFlash2WeightMemory {
            retained_bytes: add(global.retained_bytes, mul(config.num_hidden_layers, layer.retained_bytes)?)?,
            retained_buffer_count: add(global.retained_buffer_count, mul(config.num_hidden_layers, layer.retained_buffer_count)?)?,
            upload_peak_bytes: global.upload_peak_bytes.max(layer_peak),
            upload_temporary_peak_bytes: global.upload_temporary_peak_bytes.max(layer.upload_temporary_peak_bytes),
            global_upload_peak_bytes: global.upload_peak_bytes,
            layer_upload_peak_bytes: layer_peak,
        })
    }

    /// Visit allocation roots exactly once. Views and the separately owned target
    /// head are deliberately absent. Callers may inspect native resource lengths
    /// here when page-rounded allocation accounting is required.
    pub fn for_each_weight_root(&self, mut visit: impl FnMut(&MlxBuffer)) {
        let mut projection = |value: &DFlash2Projection| match value {
            DFlash2Projection::Dense { weight } => visit(weight),
            DFlash2Projection::Af4G64 { weights, metadata } => {
                visit(weights);
                visit(metadata);
            }
        };
        projection(&self.context_projection);
        projection(&self.selector_projection);
        for layer in &self.layers {
            projection(&layer.attention_dynamic);
            projection(&layer.qkv_projection);
            projection(&layer.output_projection);
            projection(&layer.mlp_dynamic);
            projection(&layer.gate_up_projection);
            projection(&layer.down_projection);
        }
        visit(&self.hidden_norm);
        visit(&self.final_norm);
        visit(&self.predecessor_codebook);
        visit(&self.successor_codebook);
        if let Some(mask) = &self.mask_embedding {
            visit(mask);
        }
        for layer in &self.layers {
            visit(&layer.attention_convolution);
            visit(&layer.input_norm);
            visit(&layer.query_norm);
            visit(&layer.key_norm);
            visit(&layer.post_attention_norm);
            visit(&layer.mlp_convolution);
        }
    }

    /// Actual loaded logical root bytes, with checked addition; never counts
    /// fused gate/up halves, context-KV aliases, or a borrowed target head twice.
    pub fn retained_weight_bytes(&self) -> Result<usize> {
        let mut total = Ok(0);
        self.for_each_weight_root(|root| {
            total = total.as_ref().map_err(|_| invalid("root byte tally overflow"))
                .and_then(|bytes| add(*bytes, root.data_byte_len()));
        });
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::config::tests::QWEN38_CONFIG;

    #[test]
    fn canonical_qwen36_payload_and_serial_upload_peak() {
        // Canonical Q36 config dimensions; remaining fields match the Q38 fixture.
        let mut config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        config.hidden_size = 2048;
        config.intermediate_size = 6144;
        config.num_hidden_layers = 6;
        config.num_target_layers = 40;
        config.target_layer_ids = vec![1, 6, 11, 16, 22, 27, 32, 37];
        config.layer_types = vec![config.layer_types[0]; 6];
        config.mask_token_id = 248077;
        for (format, retained, peak, temporary, roots) in [
            (ProjectionFormat::Bf16, 1_052_563_456, 1_077_712_896, 67_108_864, 78),
            (ProjectionFormat::Af4G64, 479_025_152, 494_310_400, 37_748_736, 116),
        ] {
            let memory = DFlash2ModelTensors::weight_memory(&config, format, false).unwrap();
            assert_eq!(
                (memory.retained_bytes, memory.retained_buffer_count,
                    memory.upload_peak_bytes, memory.upload_temporary_peak_bytes),
                (retained, roots, peak, temporary),
            );
        }
    }

    #[test]
    fn loaded_roots_match_plans_without_counting_fused_views() {
        use super::super::super::weights::{
            expected_manifest, DFlash2ExpectedTensor, DFlash2Weights, OPTIONAL_MASK_EMBEDDING,
        };
        use mlx_native::MlxDevice;
        use safetensors::tensor::{Dtype, TensorView};

        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        config.hidden_size = 64;
        config.intermediate_size = 64;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.head_dim = 64;
        config.vocab_size = 64;
        config.conv_group_size = 4;
        config.target_layer_ids = vec![0];
        config.layer_types.truncate(1);
        config.mask_token_id = 0;

        for has_mask in [false, true] {
            let mut manifest = expected_manifest(&config);
            if has_mask {
                manifest.push(DFlash2ExpectedTensor {
                    name: OPTIONAL_MASK_EMBEDDING.into(),
                    shape: vec![config.hidden_size],
                });
            }
            let data: Vec<Vec<u8>> = manifest.iter()
                .map(|entry| vec![0; entry.shape.iter().product::<usize>() * 2])
                .collect();
            let tensors = manifest.iter().zip(&data)
                .map(|(entry, bytes)| TensorView::new(Dtype::BF16, entry.shape.clone(), bytes).unwrap())
                .collect();
            let weights = DFlash2Weights { manifest, tensors };
            for format in [ProjectionFormat::Bf16, ProjectionFormat::Af4G64] {
                let plan = DFlash2ModelTensors::weight_memory(&config, format, has_mask).unwrap();
                let loaded = DFlash2ModelTensors::upload_with_format(
                    &device, &config, &weights, format,
                ).unwrap();
                assert_eq!(loaded.retained_weight_bytes().unwrap(), plan.retained_bytes);
                assert!(loaded.layers[0].gate_up_halves.is_some());
                let mut roots = 0;
                let mut resource_bytes = 0;
                loaded.for_each_weight_root(|root| {
                    roots += 1;
                    assert!(root.byte_len() >= root.data_byte_len());
                    resource_bytes += root.byte_len();
                });
                assert_eq!(roots, plan.retained_buffer_count);
                assert!(resource_bytes >= plan.retained_bytes);
                assert_eq!(loaded.mask_embedding.is_some(), has_mask);
            }
        }
    }

    #[test]
    fn canonical_qwen38_payload_and_serial_upload_peak() {
        let config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        for (format, retained, peak, temporary, roots) in [
            (ProjectionFormat::Bf16, 3_848_934_400, 4_027_151_360, 356_515_840, 66),
            (ProjectionFormat::Af4G64, 1_265_751_040, 1_396_862_976, 200_540_160, 98),
        ] {
            let memory = DFlash2ModelTensors::weight_memory(&config, format, false).unwrap();
            assert_eq!(
                (memory.retained_bytes, memory.retained_buffer_count,
                    memory.upload_peak_bytes, memory.upload_temporary_peak_bytes),
                (retained, roots, peak, temporary),
            );
            assert!(memory.upload_peak_bytes < retained + temporary);
            let masked = DFlash2ModelTensors::weight_memory(&config, format, true).unwrap();
            assert_eq!(masked.retained_bytes, retained + 2 * config.hidden_size);
            assert_eq!(masked.upload_peak_bytes, peak + 2 * config.hidden_size);
            assert_eq!(masked.retained_buffer_count, roots + 1);
        }
    }

    #[test]
    fn af4_geometry_matches_both_physical_planes_and_host_copies() {
        let memory = ProjectionMemory::new(128, 192, ProjectionFormat::Af4G64).unwrap();
        assert_eq!(memory.groups, 128 * 3);
        assert_eq!(memory.code_words, 128 * 192 / 8);
        assert_eq!(memory.retained_bytes, 128 * 192 / 2 + 128 * 3 * 4);
        assert_eq!(memory.temporary_bytes, 2 * memory.retained_bytes);
        assert_eq!(memory.roots, 2);
        assert!(ProjectionMemory::new(65, 192, ProjectionFormat::Af4G64).is_err());
        assert!(ProjectionMemory::new(128, 193, ProjectionFormat::Af4G64).is_err());
        assert!(ProjectionMemory::new(65, 193, ProjectionFormat::Bf16).is_ok());
    }

    #[test]
    fn unrepresentable_or_unsupported_shapes_fail_closed() {
        let mut config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        config.hidden_size = usize::MAX;
        assert!(DFlash2ModelTensors::weight_memory(&config, ProjectionFormat::Bf16, false).is_err());
        let mut config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        config.num_hidden_layers = usize::MAX;
        assert!(DFlash2ModelTensors::weight_memory(&config, ProjectionFormat::Af4G64, false).is_err());
        let mut config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        config.conv_group_size = 0;
        assert!(DFlash2ModelTensors::weight_memory(&config, ProjectionFormat::Bf16, false).is_err());
        assert!(ProjectionMemory::new(usize::MAX, 64, ProjectionFormat::Af4G64).is_err());
        let mut config = DFlash2Config::from_json_str(QWEN38_CONFIG).unwrap();
        config.num_attention_heads = 2;
        config.num_key_value_heads = 1;
        config.head_dim = 48;
        // The fused 192 rows are aligned, but its context-KV tail starts at 96.
        assert!(DFlash2ModelTensors::weight_memory(&config, ProjectionFormat::Af4G64, false).is_err());
        assert!(DFlash2ModelTensors::weight_memory(&config, ProjectionFormat::Bf16, false).is_ok());
    }
}
