//! The staged draft route (`HF2Q_DFLASH2_DRAFT_STAGED=1`).
//!
//! The default draft command multiplies AF4G64 planes on the register tile,
//! copies every lane's ring window into a packed K/V tile before each layer's
//! attention, and projects the vocabulary through the target's stock head
//! kernel. This route keeps the same weights and graph and changes the three
//! kernels: projections of at least [`STAGED_MIN_OUTPUTS`] outputs run the
//! staged tile over BF16 rows (gate and up in one pass), attention reads the
//! rings in place, and a width-32 head runs the prepared target tile.

use super::{
    config::DFlash2Config,
    tensors::{DFlash2ModelTensors, DFlash2Projection},
    workspace::{allocate, DFlash2Workspace, DFlash2WorkspaceError},
};
use crate::serve::forward_mlx_shared::MlxQWeight;
use anyhow::{bail, ensure, Context, Result};
use mlx_native::ops::{
    dflash2_af4g64_staged::{
        dflash2_af4g64_staged_gate_up_partial_elements, dflash2_af4g64_staged_partial_elements,
        dispatch_dflash2_af4g64_staged, dispatch_dflash2_af4g64_staged_gate_up,
    },
    dflash2_ring_attention::{
        dflash2_ring_attention_partial_elements, dflash2_ring_attention_supports,
        dispatch_dflash2_ring_attention, DFlash2RingAttentionBuffers, DFlash2RingAttentionParams,
        DFLASH2_RING_ATTENTION_CURRENT_VALUES,
    },
    elementwise::dispatch_cast_f32_to_bf16_with_encoder,
    repacked_gguf_mm::{
        dispatch_repacked_gguf_mm_bf16_input, prepare_repacked_gguf_weight, PreparedGgufWeight,
    },
};
use mlx_native::{CommandEncoder, DType, GgmlType, KernelRegistry, MlxBuffer, MlxDevice};

pub const DFLASH2_DRAFT_STAGED_ENV: &str = "HF2Q_DFLASH2_DRAFT_STAGED";

/// Narrower projections keep the register tile: a grid of fewer than one and
/// a half 64-column threadgroups per core leaves the staged tile's cores idle
/// (Splash `runtime/ops/LinearGguf.cpp:272-275`, `2 * (N / 64) * lanes <
/// 3 * cores`). The bound is fixed at four lanes on 40 cores so that a
/// projection's tile, and with it its bits, never depends on the lane count.
pub const STAGED_MIN_OUTPUTS: usize = 960;

/// Rows the prepared target head tile takes.
const PREPARED_HEAD_ROWS: usize = 32;

// The selector projection keeps the register tile and reads the F32 final
// rows, so it may run beside the head, which reads their BF16 rounding.
const _: () = assert!(mlx_native::DFLASH2_SELECTOR_RANK < STAGED_MIN_OUTPUTS);

pub(super) fn staged_tile_takes(output_width: usize) -> bool {
    output_width >= STAGED_MIN_OUTPUTS
}

/// Whether the checkpoint's shapes fit the staged route.
fn supports(config: &DFlash2Config, tensors: &DFlash2ModelTensors) -> Result<()> {
    ensure!(
        matches!(tensors.context_projection, DFlash2Projection::Af4G64 { .. }),
        "{DFLASH2_DRAFT_STAGED_ENV}=1 requires HF2Q_DFLASH2_DRAFT_AF4_G64=1"
    );
    ensure!(
        staged_tile_takes(config.hidden_size),
        "{DFLASH2_DRAFT_STAGED_ENV}=1 requires a draft hidden size of at least {STAGED_MIN_OUTPUTS}, got {}",
        config.hidden_size
    );
    ensure!(
        tensors
            .layers
            .iter()
            .all(|layer| layer.gate_up_halves.is_some()),
        "{DFLASH2_DRAFT_STAGED_ENV}=1 requires gate and up projections of whole 64-row tiles, got {} rows",
        config.intermediate_size
    );
    Ok(())
}

/// Resolve the staged-route switch against the uploaded checkpoint.
pub(super) fn resolve_staged_draft(
    raw: Option<&str>,
    config: &DFlash2Config,
    tensors: &DFlash2ModelTensors,
) -> Result<bool> {
    match raw {
        None | Some("" | "0") => Ok(false),
        Some("1") => supports(config, tensors).map(|()| true),
        Some(value) => bail!("{DFLASH2_DRAFT_STAGED_ENV} must be 0 or 1, got {value:?}"),
    }
}

/// Scratch of the ring attention.
pub struct DFlash2RingAttentionScratch {
    /// BF16 `[lanes, query heads, block, head dim]`.
    pub queries: MlxBuffer,
    /// BF16 `[lanes, KV heads, block, head dim]`.
    pub current_keys: MlxBuffer,
    /// BF16 `[lanes, KV heads, 16, head dim]`.
    pub current_values: MlxBuffer,
    pub partials: MlxBuffer,
    /// BF16 `[rows, query width]`: the rows the output projection reads.
    pub output: MlxBuffer,
}

pub struct DFlash2StagedScratch {
    /// BF16 `[rows, widest projection input]`: the rounded input of the
    /// projection being encoded.
    pub projection_input: MlxBuffer,
    /// BF16 `[rows, intermediate]`: the gated MLP activation.
    pub gated: MlxBuffer,
    /// F32 run sums of a split projection.
    pub partials: MlxBuffer,
    /// `None` when the attention shape is not the one the ring kernels are
    /// compiled for; the draft then keeps the packed attention.
    pub attention: Option<DFlash2RingAttentionScratch>,
    /// The target vocabulary head repacked for the staged tile by the first
    /// draft command that projects through it.
    pub head: Option<PreparedGgufWeight>,
}

impl DFlash2StagedScratch {
    pub(super) fn allocate(
        device: &MlxDevice,
        config: &DFlash2Config,
        lanes: usize,
        bytes: &mut usize,
    ) -> Result<Self, DFlash2WorkspaceError> {
        let block = config.block_size;
        let rows = lanes * block;
        let hidden = config.hidden_size;
        let (q_heads, kv_heads, dim) = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        let (query, key_value) = (q_heads * dim, kv_heads * dim);
        let intermediate = config.intermediate_size;
        let widest_input = config
            .fc_input_dim()
            .max(hidden)
            .max(query)
            .max(intermediate);
        let partial_elements = [
            (config.dynamic_projection_width(), hidden),
            (query + 2 * key_value, hidden),
            (hidden, query),
            (hidden, intermediate),
            (hidden, config.fc_input_dim()),
            (2 * key_value, hidden),
        ]
        .into_iter()
        .map(|(n, k)| dflash2_af4g64_staged_partial_elements(rows as u32, n as u32, k as u32))
        .chain([dflash2_af4g64_staged_gate_up_partial_elements(
            rows as u32,
            intermediate as u32,
            hidden as u32,
        )])
        .max()
        .unwrap_or(0)
        .max(1);
        let attention = if dflash2_ring_attention_supports(
            q_heads,
            kv_heads,
            dim,
            block,
            config.sliding_window,
        ) {
            Some(DFlash2RingAttentionScratch {
                queries: allocate(device, DType::BF16, &[lanes, q_heads, block, dim], bytes)?,
                current_keys: allocate(device, DType::BF16, &[lanes, kv_heads, block, dim], bytes)?,
                current_values: allocate(
                    device,
                    DType::BF16,
                    &[lanes, kv_heads, DFLASH2_RING_ATTENTION_CURRENT_VALUES, dim],
                    bytes,
                )?,
                partials: allocate(
                    device,
                    DType::F32,
                    &[dflash2_ring_attention_partial_elements(lanes, kv_heads)],
                    bytes,
                )?,
                output: allocate(device, DType::BF16, &[rows, query], bytes)?,
            })
        } else {
            None
        };
        Ok(Self {
            projection_input: allocate(device, DType::BF16, &[rows, widest_input], bytes)?,
            gated: allocate(device, DType::BF16, &[rows, intermediate], bytes)?,
            partials: allocate(device, DType::F32, &[partial_elements], bytes)?,
            attention,
            head: None,
        })
    }

    /// Round F32 rows into `projection_input`.
    pub(super) fn round_input(
        &self,
        encoder: &mut CommandEncoder,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        dispatch_cast_f32_to_bf16_with_encoder(
            encoder,
            registry,
            device.metal_device(),
            input,
            &self.projection_input,
            u32::try_from(rows * width)?,
        )?;
        encoder.memory_barrier();
        Ok(())
    }

    /// Project BF16 rows through AF4G64 planes on the staged tile.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        &self,
        encoder: &mut CommandEncoder,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        weights: &MlxBuffer,
        metadata: &MlxBuffer,
        input: &MlxBuffer,
        output: &MlxBuffer,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        dispatch_dflash2_af4g64_staged(
            encoder,
            registry,
            device,
            input,
            weights,
            metadata,
            &self.partials,
            output,
            u32::try_from(rows)?,
            u32::try_from(output_width)?,
            u32::try_from(input_width)?,
        )
        .map_err(anyhow::Error::from)
    }

    /// `gated = SiLU(projection_input x gate^T) * (projection_input x up^T)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gate_up(
        &self,
        encoder: &mut CommandEncoder,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        halves: &[DFlash2Projection; 2],
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let [DFlash2Projection::Af4G64 {
            weights: gate_weights,
            metadata: gate_metadata,
        }, DFlash2Projection::Af4G64 {
            weights: up_weights,
            metadata: up_metadata,
        }] = halves
        else {
            bail!("the staged gate/up pair requires AF4G64 planes");
        };
        dispatch_dflash2_af4g64_staged_gate_up(
            encoder,
            registry,
            device,
            &self.projection_input,
            gate_weights,
            gate_metadata,
            up_weights,
            up_metadata,
            &self.partials,
            &self.gated,
            u32::try_from(rows)?,
            u32::try_from(output_width)?,
            u32::try_from(input_width)?,
        )
        .map_err(anyhow::Error::from)
    }
}

/// Attention of the rotated current rows over layer `layer_index`'s rings.
pub(super) fn dispatch_ring_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    workspace: &DFlash2Workspace,
    scratch: &DFlash2RingAttentionScratch,
    layer_index: usize,
) -> Result<()> {
    dispatch_dflash2_ring_attention(
        encoder,
        registry,
        device,
        &DFlash2RingAttentionBuffers {
            query: &workspace.query,
            key: &workspace.key,
            value: &workspace.value,
            ring_keys: &workspace.ring_keys[layer_index],
            ring_values: &workspace.ring_values[layer_index],
            cache_lengths: &workspace.cache_lengths,
            queries: &scratch.queries,
            current_keys: &scratch.current_keys,
            current_values: &scratch.current_values,
            partials: &scratch.partials,
            output: &scratch.output,
        },
        DFlash2RingAttentionParams {
            lanes: u32::try_from(workspace.max_lanes)?,
            q_heads: u32::try_from(config.num_attention_heads)?,
            kv_heads: u32::try_from(config.num_key_value_heads)?,
            window: u32::try_from(config.sliding_window)?,
        },
    )
    .map_err(anyhow::Error::from)
}

/// Project `input` through the target's native head on the prepared tile.
/// Returns `false`, encoding nothing, when the head or the row count is not
/// one the tile takes.
pub(super) fn dispatch_prepared_head(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    staged: &mut DFlash2StagedScratch,
    weight: &MlxQWeight,
    input: &MlxBuffer,
    output: &MlxBuffer,
    rows: usize,
) -> Result<bool> {
    if rows != PREPARED_HEAD_ROWS
        || weight.affine.is_some()
        || !matches!(weight.info.ggml_dtype, GgmlType::Q4_K | GgmlType::Q6_K)
        || weight.info.rows % 64 != 0
        || weight.info.cols % 256 != 0
    {
        return Ok(false);
    }
    let params = weight.matmul_params(u32::try_from(rows)?)?;
    if staged.head.is_none() {
        staged.head = Some(
            prepare_repacked_gguf_weight(encoder, registry, device, &weight.buffer, &params)
                .context("prepare the DFlash2 vocabulary head")?,
        );
    }
    let prepared = staged.head.as_ref().expect("prepared above");
    staged.round_input(encoder, registry, device, input, rows, weight.info.cols)?;
    dispatch_repacked_gguf_mm_bf16_input(
        encoder,
        registry,
        device,
        &staged.projection_input,
        prepared,
        output,
        &params,
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash2::{
        config::DFlash2LayerType,
        forward::{
            dispatch_dflash2_context_commit, dispatch_dflash2_forward, prepare_decode_positions,
            DFlash2VocabularyHead,
        },
    };
    use half::bf16;

    /// The smallest shape every staged kernel takes: 16 query heads over 4 KV
    /// heads of 128, a 256-slot ring, and projections of 1024 outputs or more.
    fn config(hidden: usize) -> DFlash2Config {
        DFlash2Config {
            draft_model_type: "qwen3".into(),
            hidden_size: hidden,
            intermediate_size: hidden,
            num_hidden_layers: 2,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            vocab_size: 64,
            num_target_layers: 1,
            target_layer_ids: vec![0],
            layer_types: vec![DFlash2LayerType::SlidingAttention; 2],
            max_position_embeddings: 4_096,
            sliding_window: 256,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            mask_token_id: 0,
            final_logit_softcapping: None,
            output_multiplier: None,
        }
    }

    #[test]
    fn the_switch_is_off_unless_exactly_one_and_fails_closed() {
        let device = MlxDevice::new().unwrap();
        let cfg = config(1_024);
        let mut tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &cfg, 3);
        for off in [None, Some(""), Some("0")] {
            assert!(!resolve_staged_draft(off, &cfg, &tensors).unwrap());
        }
        assert!(resolve_staged_draft(Some("1"), &cfg, &tensors).unwrap());
        for invalid in ["true", "on", " 1", "2"] {
            let error = resolve_staged_draft(Some(invalid), &cfg, &tensors).unwrap_err();
            assert!(error.to_string().contains("must be 0 or 1"), "{error}");
        }

        let narrow = config(512);
        let narrow_tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &narrow, 3);
        let error = resolve_staged_draft(Some("1"), &narrow, &narrow_tensors).unwrap_err();
        assert!(error.to_string().contains("hidden size"), "{error}");
        assert!(!resolve_staged_draft(Some("0"), &narrow, &narrow_tensors).unwrap());

        tensors.layers[0].gate_up_halves = None;
        let error = resolve_staged_draft(Some("1"), &cfg, &tensors).unwrap_err();
        assert!(error.to_string().contains("64-row tiles"), "{error}");

        tensors.context_projection = DFlash2Projection::Dense {
            weight: device.alloc_buffer(2, DType::BF16, vec![1, 1]).unwrap(),
        };
        let error = resolve_staged_draft(Some("1"), &cfg, &tensors).unwrap_err();
        assert!(
            error.to_string().contains("HF2Q_DFLASH2_DRAFT_AF4_G64=1"),
            "{error}"
        );
    }

    fn random(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
    }

    struct Outputs {
        final_hidden: Vec<f32>,
        logits: Vec<f32>,
        selector_hidden: Vec<f32>,
        rings: Vec<f32>,
    }

    /// Fill the rings past one wrap, then run one draft forward.
    fn run(lanes: usize, staged: bool) -> Outputs {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let cfg = config(1_024);
        let tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &cfg, 5);
        assert!(resolve_staged_draft(Some("1"), &cfg, &tensors).unwrap());
        let mut head = device
            .alloc_buffer(
                cfg.vocab_size * cfg.hidden_size * 2,
                DType::BF16,
                vec![cfg.vocab_size, cfg.hidden_size],
            )
            .unwrap();
        let mut state = 17_u64;
        for weight in head.as_mut_slice::<bf16>().unwrap() {
            *weight = bf16::from_f32(0.05 * random(&mut state));
        }
        let mut workspace = DFlash2Workspace::new(&device, &cfg, lanes, staged).unwrap();
        assert_eq!(workspace.staged.is_some(), staged);
        let retained_rows: Vec<u32> = [8_u32, 8, 3, 0][..lanes].to_vec();
        let mut retained = device
            .alloc_buffer(lanes * 4, DType::U32, vec![lanes])
            .unwrap();
        retained
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&retained_rows);
        let mut lengths = vec![0_u32; lanes];
        for _ in 0..40 {
            for value in workspace
                .captured_target_hidden
                .as_mut_slice::<f32>()
                .unwrap()
            {
                *value = 2.0 * random(&mut state);
            }
            prepare_decode_positions(&mut workspace, &cfg, &lengths).unwrap();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_dflash2_context_commit(
                &mut encoder,
                &mut registry,
                &device,
                &cfg,
                &tensors,
                &retained,
                &workspace,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            for (length, rows) in lengths.iter_mut().zip(&retained_rows) {
                *length += rows;
            }
        }
        prepare_decode_positions(&mut workspace, &cfg, &lengths).unwrap();
        for value in workspace.hidden[0].as_mut_slice::<f32>().unwrap() {
            *value = 2.0 * random(&mut state);
        }
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_forward(
            &mut encoder,
            &mut registry,
            &device,
            &cfg,
            &tensors,
            DFlash2VocabularyHead::DenseBf16(&head),
            &mut workspace,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        Outputs {
            final_hidden: workspace.final_hidden.as_slice::<f32>().unwrap().to_vec(),
            logits: workspace.logits.as_slice::<f32>().unwrap().to_vec(),
            selector_hidden: workspace
                .selector_hidden_f32
                .as_slice::<f32>()
                .unwrap()
                .to_vec(),
            rings: workspace
                .ring_keys
                .iter()
                .chain(&workspace.ring_values)
                .flat_map(|ring| ring.as_slice::<bf16>().unwrap())
                .map(|value| value.to_f32())
                .collect(),
        }
    }

    fn relative_rms(actual: &[f32], expected: &[f32]) -> f64 {
        assert_eq!(actual.len(), expected.len());
        let (mut error, mut magnitude) = (0.0_f64, 0.0_f64);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            error += f64::from(actual - expected).powi(2);
            magnitude += f64::from(*expected).powi(2);
        }
        assert!(magnitude > 0.0);
        (error / magnitude).sqrt()
    }

    /// The staged route computes the default route's graph: same rings, same
    /// hidden states and same logits up to the kernels' rounding, at every
    /// lane count's row tiling.
    #[test]
    fn staged_route_matches_the_default_route_on_a_synthetic_checkpoint() {
        for lanes in [1, 3, 4] {
            let default = run(lanes, false);
            let staged = run(lanes, true);
            for (name, actual, expected) in [
                ("rings", &staged.rings, &default.rings),
                ("final hidden", &staged.final_hidden, &default.final_hidden),
                (
                    "selector hidden",
                    &staged.selector_hidden,
                    &default.selector_hidden,
                ),
                ("logits", &staged.logits, &default.logits),
            ] {
                let error = relative_rms(actual, expected);
                assert!(
                    error <= 5.0e-3,
                    "{lanes} lanes: staged {name} differs by relative RMS {error}"
                );
            }
            // The routes are distinct kernels, not one route run twice.
            assert_ne!(
                staged
                    .final_hidden
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                default
                    .final_hidden
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}
