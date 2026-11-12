//! GPU-resident DFlash2 weights shared by every target-family adapter.

use super::config::DFlash2Config;
use super::route_profile;
use super::weights::{DFlash2Weights, OPTIONAL_MASK_EMBEDDING};
use half::bf16;
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use safetensors::tensor::TensorView;

pub const DFLASH2_DRAFT_AF4_G64_ENV: &str = "HF2Q_DFLASH2_DRAFT_AF4_G64";

#[derive(Debug, thiserror::Error)]
pub enum DFlash2TensorsError {
    #[error("DFlash2 tensor missing after schema validation: {0}")]
    Missing(String),
    #[error("DFlash2 tensor upload: {0}")]
    Mlx(#[from] MlxError),
    #[error("DFlash2 projection quantization: {0}")]
    Quantization(String),
}

pub enum DFlash2Projection {
    Dense { weight: MlxBuffer },
    Af4G64 {
        weights: MlxBuffer,
        metadata: MlxBuffer,
    },
}

impl DFlash2Projection {
    pub fn format_label(&self) -> &'static str {
        match self {
            Self::Dense { .. } => "bf16",
            Self::Af4G64 { .. } => "af4_g64",
        }
    }
}

#[derive(Clone, Copy)]
enum ProjectionFormat {
    Bf16,
    Af4G64,
}

pub struct DFlash2LayerTensors {
    pub attention_dynamic: DFlash2Projection,
    pub attention_convolution: MlxBuffer,
    pub input_norm: MlxBuffer,
    pub qkv_projection: DFlash2Projection,
    pub context_kv_projection: DFlash2Projection,
    pub output_projection: DFlash2Projection,
    pub query_norm: MlxBuffer,
    pub key_norm: MlxBuffer,
    pub mlp_dynamic: DFlash2Projection,
    pub post_attention_norm: MlxBuffer,
    pub gate_up_projection: DFlash2Projection,
    /// The gate and up rows of `gate_up_projection` as views, when both
    /// start at a storage tile of its format.
    pub gate_up_halves: Option<[DFlash2Projection; 2]>,
    pub down_projection: DFlash2Projection,
    pub mlp_convolution: MlxBuffer,
}

pub struct DFlash2ModelTensors {
    pub context_projection: DFlash2Projection,
    pub hidden_norm: MlxBuffer,
    pub final_norm: MlxBuffer,
    pub selector_projection: DFlash2Projection,
    pub predecessor_codebook: MlxBuffer,
    pub successor_codebook: MlxBuffer,
    pub mask_embedding: Option<MlxBuffer>,
    pub layers: Vec<DFlash2LayerTensors>,
}

fn tensor<'view, 'data>(
    weights: &'view DFlash2Weights<'data>,
    name: &str,
) -> Result<&'view TensorView<'data>, DFlash2TensorsError> {
    weights
        .tensor(name)
        .ok_or_else(|| DFlash2TensorsError::Missing(name.into()))
}

fn upload_bf16(
    device: &MlxDevice,
    view: &TensorView<'_>,
) -> Result<MlxBuffer, DFlash2TensorsError> {
    let mut buffer = device.alloc_buffer(view.data().len(), DType::BF16, view.shape().to_vec())?;
    buffer.as_mut_slice::<u8>()?.copy_from_slice(view.data());
    Ok(buffer)
}

fn projection_geometry(view: &TensorView<'_>) -> Result<(usize, usize), DFlash2TensorsError> {
    let (&columns, outer) = view
        .shape()
        .split_last()
        .ok_or_else(|| DFlash2TensorsError::Quantization("projection has no dimensions".into()))?;
    let rows = outer.iter().try_fold(1usize, |rows, dim| {
        rows.checked_mul(*dim)
            .ok_or_else(|| DFlash2TensorsError::Quantization("projection rows overflow".into()))
    })?;
    let expected_bytes = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| DFlash2TensorsError::Quantization("projection bytes overflow".into()))?;
    if rows == 0 || columns == 0 || view.data().len() != expected_bytes {
        return Err(DFlash2TensorsError::Quantization(format!(
            "projection shape {:?} requires {expected_bytes} BF16 bytes, found {}",
            view.shape(),
            view.data().len()
        )));
    }
    Ok((rows, columns))
}

fn append_af4_g64(
    view: &TensorView<'_>,
    codes: &mut Vec<u32>,
    scales: &mut Vec<u16>,
    biases: &mut Vec<u16>,
) -> Result<(), DFlash2TensorsError> {
    const GROUP: usize = 64;
    let (_, columns) = projection_geometry(view)?;
    if columns % GROUP != 0 {
        return Err(DFlash2TensorsError::Quantization(format!(
            "AF4G64 projection width {columns} is not a multiple of {GROUP}"
        )));
    }
    for group in view.data().chunks_exact(GROUP * 2) {
        let mut values = [0.0f32; GROUP];
        for (value, bytes) in values.iter_mut().zip(group.chunks_exact(2)) {
            *value = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
            if !value.is_finite() {
                return Err(DFlash2TensorsError::Quantization(
                    "AF4G64 projection contains a non-finite weight".into(),
                ));
            }
        }
        let mut minimum = values[0];
        let mut maximum = values[0];
        for value in values.iter().copied() {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        maximum = maximum.max(0.0);
        let minimum_edge = minimum.abs() > maximum.abs();
        let mut step = ((maximum - minimum) / 15.0).max(1.0e-7);
        if !minimum_edge {
            step = -step;
        }
        let edge = if minimum_edge { minimum } else { maximum };
        let q0 = (edge / step).round();
        let offset = if q0 != 0.0 {
            step = edge / q0;
            edge
        } else {
            0.0
        };
        scales.push(bf16::from_f32(step).to_bits());
        biases.push(bf16::from_f32(offset).to_bits());
        for chunk in values.chunks_exact(8) {
            let mut packed = 0u32;
            for (index, value) in chunk.iter().copied().enumerate() {
                let code = ((value - offset) / step).round().clamp(0.0, 15.0) as u32;
                packed |= code << (index * 4);
            }
            codes.push(packed);
        }
    }
    Ok(())
}

fn prepare_af4_g64_planes(
    rows: usize,
    columns: usize,
    row_words: &[u32],
    scales: &[u16],
    biases: &[u16],
) -> Result<(Vec<u32>, Vec<u32>), DFlash2TensorsError> {
    const TILE_ROWS: usize = 64;
    if rows % TILE_ROWS != 0 {
        return Err(DFlash2TensorsError::Quantization(format!(
            "AF4G64 projection rows {rows} are not a multiple of {TILE_ROWS}"
        )));
    }
    let groups32 = columns / 32;
    let groups64 = columns / 64;
    let mut plane = vec![0u32; rows * groups32 * 4];
    let mut metadata = vec![0u32; rows * groups64];
    for row in 0..rows {
        for group64 in 0..groups64 {
            let source_metadata = row * groups64 + group64;
            let metadata_at =
                (row / TILE_ROWS * groups64 + group64) * TILE_ROWS + row % TILE_ROWS;
            metadata[metadata_at] =
                u32::from(scales[source_metadata]) | (u32::from(biases[source_metadata]) << 16);
            for group_half in 0..2 {
                let group32 = group64 * 2 + group_half;
                for chunk in 0..4 {
                    let mut word = 0u32;
                    for pair in 0..4 {
                        let first =
                            group32 * 32 + 16 * (pair >> 1) + 4 * chunk + 2 * (pair & 1);
                        let code = |column: usize| {
                            let packed = row_words[row * (columns / 8) + column / 8];
                            (packed >> (4 * (column % 8))) & 0x0f
                        };
                        word |= code(first) << (4 * pair);
                        word |= code(first + 1) << (16 + 4 * pair);
                    }
                    let at = ((row / TILE_ROWS * groups32 + group32) * TILE_ROWS
                        + row % TILE_ROWS)
                        * 4
                        + chunk;
                    plane[at] = word;
                }
            }
        }
    }
    Ok((plane, metadata))
}

fn upload_fused_projection(
    device: &MlxDevice,
    views: &[&TensorView<'_>],
    format: ProjectionFormat,
) -> Result<DFlash2Projection, DFlash2TensorsError> {
    let first = views
        .first()
        .ok_or_else(|| DFlash2TensorsError::Quantization("empty projection fusion".into()))?;
    let (_, columns) = projection_geometry(first)?;
    let mut rows = 0usize;
    for view in views {
        let (view_rows, view_columns) = projection_geometry(view)?;
        if view_columns != columns {
            return Err(DFlash2TensorsError::Quantization(format!(
                "projection fusion width {view_columns} != {columns}"
            )));
        }
        rows = rows
            .checked_add(view_rows)
            .ok_or_else(|| DFlash2TensorsError::Quantization("fused rows overflow".into()))?;
    }

    if matches!(format, ProjectionFormat::Af4G64) {
        let groups = rows
            .checked_mul(columns / 64)
            .ok_or_else(|| DFlash2TensorsError::Quantization("AF4G64 groups overflow".into()))?;
        let mut code_words = Vec::with_capacity(groups * 8);
        let mut scale_bits = Vec::with_capacity(groups);
        let mut bias_bits = Vec::with_capacity(groups);
        for view in views {
            append_af4_g64(view, &mut code_words, &mut scale_bits, &mut bias_bits)?;
        }
        let (prepared_words, metadata_words) =
            prepare_af4_g64_planes(rows, columns, &code_words, &scale_bits, &bias_bits)?;
        let mut weights = device.alloc_buffer(
            prepared_words.len() * 4,
            DType::U32,
            vec![rows / 64, columns / 32, 64, 4],
        )?;
        weights
            .as_mut_slice::<u32>()?
            .copy_from_slice(&prepared_words);
        let mut metadata = device.alloc_buffer(
            metadata_words.len() * 4,
            DType::U32,
            vec![rows / 64, columns / 64, 64],
        )?;
        metadata
            .as_mut_slice::<u32>()?
            .copy_from_slice(&metadata_words);
        return Ok(DFlash2Projection::Af4G64 { weights, metadata });
    }

    let mut bytes = Vec::new();
    for view in views {
        bytes.extend_from_slice(view.data());
    }
    let mut weight = device.alloc_buffer(bytes.len(), DType::BF16, vec![rows, columns])?;
    weight.as_mut_slice::<u8>()?.copy_from_slice(&bytes);
    Ok(DFlash2Projection::Dense { weight })
}

fn upload_projection(
    device: &MlxDevice,
    view: &TensorView<'_>,
    format: ProjectionFormat,
) -> Result<DFlash2Projection, DFlash2TensorsError> {
    upload_fused_projection(device, &[view], format)
}

fn buffer_row_view(
    buffer: &MlxBuffer,
    total_rows: usize,
    first_row: usize,
    rows: usize,
) -> Result<MlxBuffer, DFlash2TensorsError> {
    let row_bytes = buffer
        .byte_len()
        .checked_div(total_rows)
        .ok_or_else(|| DFlash2TensorsError::Quantization("projection has zero rows".into()))?;
    if row_bytes.checked_mul(total_rows) != Some(buffer.byte_len())
        || first_row
            .checked_add(rows)
            .is_none_or(|end| end > total_rows)
    {
        return Err(DFlash2TensorsError::Quantization(
            "projection row view is not physically aligned".into(),
        ));
    }
    let byte_offset = first_row
        .checked_mul(row_bytes)
        .ok_or_else(|| DFlash2TensorsError::Quantization("row offset overflow".into()))?;
    let byte_len = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| DFlash2TensorsError::Quantization("row length overflow".into()))?;
    let element_bytes = buffer.dtype().size_of();
    if byte_len % element_bytes != 0 {
        return Err(DFlash2TensorsError::Quantization(
            "projection row view splits a storage element".into(),
        ));
    }
    let byte_offset = u64::try_from(byte_offset)
        .map_err(|_| DFlash2TensorsError::Quantization("row offset exceeds u64".into()))?;
    Ok(buffer.slice_view(byte_offset, byte_len / element_bytes))
}

fn projection_row_view(
    projection: &DFlash2Projection,
    total_rows: usize,
    first_row: usize,
    rows: usize,
) -> Result<DFlash2Projection, DFlash2TensorsError> {
    match projection {
        DFlash2Projection::Dense { weight } => Ok(DFlash2Projection::Dense {
            weight: buffer_row_view(weight, total_rows, first_row, rows)?,
        }),
        DFlash2Projection::Af4G64 { weights, metadata } => {
            if first_row % 64 != 0 || rows % 64 != 0 {
                return Err(DFlash2TensorsError::Quantization(format!(
                    "AF4G64 projection row view {first_row}..{} is not 64-row aligned",
                    first_row + rows
                )));
            }
            Ok(DFlash2Projection::Af4G64 {
                weights: buffer_row_view(weights, total_rows, first_row, rows)?,
                metadata: buffer_row_view(metadata, total_rows, first_row, rows)?,
            })
        }
    }
}

fn upload_norm_f32(
    device: &MlxDevice,
    view: &TensorView<'_>,
) -> Result<MlxBuffer, DFlash2TensorsError> {
    let mut buffer =
        device.alloc_buffer(view.data().len() * 2, DType::F32, view.shape().to_vec())?;
    let output = buffer.as_mut_slice::<f32>()?;
    for (value, bytes) in output.iter_mut().zip(view.data().chunks_exact(2)) {
        *value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
    }
    Ok(buffer)
}

impl DFlash2ModelTensors {
    pub fn upload(
        device: &MlxDevice,
        config: &DFlash2Config,
        weights: &DFlash2Weights<'_>,
    ) -> Result<Self, DFlash2TensorsError> {
        const AF4_G64: route_profile::Switch = route_profile::switch(DFLASH2_DRAFT_AF4_G64_ENV);
        let format = if route_profile::is_one(AF4_G64) {
            ProjectionFormat::Af4G64
        } else {
            ProjectionFormat::Bf16
        };
        let context_projection = upload_projection(device, tensor(weights, "fc.weight")?, format)?;
        let hidden_norm = upload_norm_f32(device, tensor(weights, "hidden_norm.weight")?)?;
        let final_norm = upload_norm_f32(device, tensor(weights, "norm.weight")?)?;
        let selector_projection = upload_projection(
            device,
            tensor(weights, "candidate_selector.hidden_projection.weight")?,
            format,
        )?;
        let predecessor_codebook = upload_bf16(
            device,
            tensor(weights, "candidate_selector.predecessor_codebook")?,
        )?;
        let successor_codebook = upload_bf16(
            device,
            tensor(weights, "candidate_selector.successor_codebook")?,
        )?;
        let mask_embedding = weights
            .tensor(OPTIONAL_MASK_EMBEDDING)
            .map(|view| upload_bf16(device, view))
            .transpose()?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            let bf16 =
                |suffix: &str| upload_bf16(device, tensor(weights, &format!("{prefix}.{suffix}"))?);
            let projection = |suffix: &str| {
                upload_projection(
                    device,
                    tensor(weights, &format!("{prefix}.{suffix}"))?,
                    format,
                )
            };
            let norm = |suffix: &str| {
                upload_norm_f32(device, tensor(weights, &format!("{prefix}.{suffix}"))?)
            };
            let q = tensor(weights, &format!("{prefix}.self_attn.q_proj.weight"))?;
            let k = tensor(weights, &format!("{prefix}.self_attn.k_proj.weight"))?;
            let v = tensor(weights, &format!("{prefix}.self_attn.v_proj.weight"))?;
            let (q_rows, q_columns) = projection_geometry(q)?;
            let (k_rows, k_columns) = projection_geometry(k)?;
            let (v_rows, v_columns) = projection_geometry(v)?;
            if q_columns != k_columns || q_columns != v_columns {
                return Err(DFlash2TensorsError::Quantization(format!(
                    "{prefix} Q/K/V projection input widths differ"
                )));
            }
            let qkv_projection = upload_fused_projection(device, &[q, k, v], format)?;
            let qkv_rows = q_rows
                .checked_add(k_rows)
                .and_then(|rows| rows.checked_add(v_rows))
                .ok_or_else(|| {
                    DFlash2TensorsError::Quantization("Q/K/V fused rows overflow".into())
                })?;
            let context_kv_projection =
                projection_row_view(&qkv_projection, qkv_rows, q_rows, k_rows + v_rows)?;
            let gate = tensor(weights, &format!("{prefix}.mlp.gate_proj.weight"))?;
            let up = tensor(weights, &format!("{prefix}.mlp.up_proj.weight"))?;
            let gate_up_projection = upload_fused_projection(device, &[gate, up], format)?;
            let (gate_rows, _) = projection_geometry(gate)?;
            let (up_rows, _) = projection_geometry(up)?;
            let half = |first_row, rows| {
                projection_row_view(&gate_up_projection, gate_rows + up_rows, first_row, rows).ok()
            };
            let gate_up_halves = match (half(0, gate_rows), half(gate_rows, up_rows)) {
                (Some(gate), Some(up)) if gate_rows == up_rows => Some([gate, up]),
                _ => None,
            };
            layers.push(DFlash2LayerTensors {
                attention_dynamic: projection("attention_conv.kernel_projection.weight")?,
                attention_convolution: bf16("attention_conv.base_kernel")?,
                input_norm: norm("input_layernorm.weight")?,
                qkv_projection,
                context_kv_projection,
                output_projection: projection("self_attn.o_proj.weight")?,
                query_norm: norm("self_attn.q_norm.weight")?,
                key_norm: norm("self_attn.k_norm.weight")?,
                mlp_dynamic: projection("mlp_conv.kernel_projection.weight")?,
                post_attention_norm: norm("post_attention_layernorm.weight")?,
                gate_up_projection,
                gate_up_halves,
                down_projection: projection("mlp.down_proj.weight")?,
                mlp_convolution: bf16("mlp_conv.base_kernel")?,
            });
        }
        Ok(Self {
            context_projection,
            hidden_norm,
            final_norm,
            selector_projection,
            predecessor_codebook,
            successor_codebook,
            mask_embedding,
            layers,
        })
    }
}

#[cfg(test)]
impl DFlash2ModelTensors {
    /// Seeded AF4G64 tensors of `config`'s shapes with unit norm weights, for
    /// tests that compare draft routes on one checkpoint.
    pub(super) fn synthetic_af4_g64(device: &MlxDevice, config: &DFlash2Config, seed: u64) -> Self {
        use safetensors::Dtype;
        let mut state = seed;
        let mut bf16_bytes = |count: usize, scale: f32| -> Vec<u8> {
            (0..count)
                .flat_map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    bf16::from_f32(2.0 * scale * unit).to_le_bytes()
                })
                .collect()
        };
        let mut projection = |parts: &[usize], columns: usize| {
            let scale = 1.0 / (columns as f32).sqrt();
            let bytes: Vec<Vec<u8>> = parts
                .iter()
                .map(|rows| bf16_bytes(rows * columns, scale))
                .collect();
            let views: Vec<TensorView<'_>> = parts
                .iter()
                .zip(&bytes)
                .map(|(rows, bytes)| {
                    TensorView::new(Dtype::BF16, vec![*rows, columns], bytes).unwrap()
                })
                .collect();
            upload_fused_projection(
                device,
                &views.iter().collect::<Vec<_>>(),
                ProjectionFormat::Af4G64,
            )
            .unwrap()
        };
        let ones = |width: usize| {
            let mut buffer = device
                .alloc_buffer(width * 4, DType::F32, vec![width])
                .unwrap();
            buffer.as_mut_slice::<f32>().unwrap().fill(1.0);
            buffer
        };
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query = config.num_attention_heads * config.head_dim;
        let key_value = config.num_key_value_heads * config.head_dim;
        let dynamic = config.dynamic_projection_width();
        let layers = (0..config.num_hidden_layers)
            .map(|_| {
                let qkv_projection = projection(&[query, key_value, key_value], hidden);
                let context_kv_projection = projection_row_view(
                    &qkv_projection,
                    query + 2 * key_value,
                    query,
                    2 * key_value,
                )
                .unwrap();
                let gate_up_projection = projection(&[intermediate, intermediate], hidden);
                let half = |first_row| {
                    projection_row_view(
                        &gate_up_projection,
                        2 * intermediate,
                        first_row,
                        intermediate,
                    )
                    .unwrap()
                };
                let gate_up_halves = Some([half(0), half(intermediate)]);
                DFlash2LayerTensors {
                    attention_dynamic: projection(&[dynamic], hidden),
                    attention_convolution: device
                        .alloc_buffer(8 * hidden, DType::BF16, vec![2, 2, hidden])
                        .unwrap(),
                    input_norm: ones(hidden),
                    qkv_projection,
                    context_kv_projection,
                    output_projection: projection(&[hidden], query),
                    query_norm: ones(config.head_dim),
                    key_norm: ones(config.head_dim),
                    mlp_dynamic: projection(&[dynamic], hidden),
                    post_attention_norm: ones(hidden),
                    gate_up_projection,
                    gate_up_halves,
                    down_projection: projection(&[hidden], intermediate),
                    mlp_convolution: device
                        .alloc_buffer(8 * hidden, DType::BF16, vec![2, 2, hidden])
                        .unwrap(),
                }
            })
            .collect();
        let codebook = || {
            device
                .alloc_buffer(
                    config.vocab_size * config.selector_rank * 2,
                    DType::BF16,
                    vec![config.vocab_size, config.selector_rank],
                )
                .unwrap()
        };
        Self {
            context_projection: projection(&[hidden], config.fc_input_dim()),
            hidden_norm: ones(hidden),
            final_norm: ones(hidden),
            selector_projection: projection(&[config.selector_rank], hidden),
            predecessor_codebook: codebook(),
            successor_codebook: codebook(),
            mask_embedding: None,
            layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype;

    #[test]
    fn bf16_norm_decode_is_exact() {
        let bytes = [0x80, 0x3f, 0x00, 0xc0];
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect();
        assert_eq!(values, [1.0, -2.0]);
    }

    #[test]
    fn af4_g64_matches_affine_group_layout() {
        let bytes: Vec<u8> = (0..64)
            .flat_map(|index| bf16::from_f32((index % 16) as f32).to_le_bytes())
            .collect();
        let view = TensorView::new(Dtype::BF16, vec![1, 64], &bytes).unwrap();
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();

        append_af4_g64(&view, &mut codes, &mut scales, &mut biases).unwrap();

        assert_eq!(
            codes,
            [
                0x89ab_cdef,
                0x0123_4567,
                0x89ab_cdef,
                0x0123_4567,
                0x89ab_cdef,
                0x0123_4567,
                0x89ab_cdef,
                0x0123_4567,
            ]
        );
        assert_eq!(scales, [bf16::from_f32(-1.0).to_bits()]);
        assert_eq!(biases, [bf16::from_f32(15.0).to_bits()]);
        assert_eq!(codes.len() * 4 + scales.len() * 2 + biases.len() * 2, 36);
    }
}
