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

/// Whether a draft command over `workspace` projects its vocabulary through
/// prepared head planes: the staged route, at the prepared tile's row count.
/// Only such a command reads a prepared head, its own or the target's.
pub(super) fn reads_prepared_head(workspace: &DFlash2Workspace, config: &DFlash2Config) -> bool {
    workspace.staged.is_some() && workspace.max_lanes * config.block_size == PREPARED_HEAD_ROWS
}

/// Rows of the smallest staged tile; a command's rows are covered by 32-,
/// 16- and 8-row tiles.
const STAGED_TILE_ROWS: usize = 8;

// The selector projection keeps the register tile and reads the F32 final
// rows, so it may run beside the head, which reads their BF16 rounding.
const _: () = assert!(mlx_native::DFLASH2_SELECTOR_RANK < STAGED_MIN_OUTPUTS);

pub(super) fn staged_tile_takes(output_width: usize) -> bool {
    output_width >= STAGED_MIN_OUTPUTS
}

/// Whether the checkpoint's shapes at `lanes` lanes fit the staged route.
fn supports(config: &DFlash2Config, tensors: &DFlash2ModelTensors, lanes: usize) -> Result<()> {
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
    ensure!(
        (lanes * config.block_size) % STAGED_TILE_ROWS == 0,
        "{DFLASH2_DRAFT_STAGED_ENV}=1 requires a multiple of {STAGED_TILE_ROWS} draft rows, got {lanes} lanes of {} rows",
        config.block_size
    );
    Ok(())
}

/// Resolve the staged-route switch against the uploaded checkpoint.
pub(super) fn resolve_staged_draft(
    raw: Option<&str>,
    config: &DFlash2Config,
    tensors: &DFlash2ModelTensors,
    lanes: usize,
) -> Result<bool> {
    match raw {
        None | Some("" | "0") => Ok(false),
        Some("1") => supports(config, tensors, lanes).map(|()| true),
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
    /// The target vocabulary head repacked for the staged tile. It is
    /// recorded by [`Self::command_committed`], once the command that repacks
    /// it has been committed: a command abandoned while it is encoded leaves
    /// no head whose planes were never written.
    pub head: Option<PreparedGgufWeight>,
    /// The head the command being encoded repacks.
    pending_head: Option<PreparedGgufWeight>,
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
                    &[dflash2_ring_attention_partial_elements(lanes, kv_heads)?],
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
            pending_head: None,
        })
    }

    /// Record that the command encoded through this scratch was committed.
    pub fn command_committed(&mut self) {
        if let Some(head) = self.pending_head.take() {
            self.head = Some(head);
        }
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
/// one the tile takes. With `shared`, the planes are the target's own and
/// nothing is repacked here. Otherwise the first command also repacks the
/// head; the caller reports that command's commit with
/// [`DFlash2StagedScratch::command_committed`].
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_prepared_head(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    staged: &mut DFlash2StagedScratch,
    weight: &MlxQWeight,
    shared: Option<&PreparedGgufWeight>,
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
    if shared.is_none() && staged.head.is_none() {
        // Replaces a head that a command abandoned before its commit prepared.
        staged.pending_head = Some(
            prepare_repacked_gguf_weight(encoder, registry, device, &weight.buffer, &params)
                .context("prepare the DFlash2 vocabulary head")?,
        );
    }
    let prepared = shared
        .or(staged.head.as_ref())
        .or(staged.pending_head.as_ref())
        .expect("prepared above");
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
        adapters::{target_vocabulary_head, DFlash2TargetFamily},
        config::{DFlash2LayerType, DFlash2TargetShape},
        forward::{
            dispatch_dflash2_context_commit, dispatch_dflash2_forward, prepare_decode_positions,
            DFlash2VocabularyHead,
        },
    };
    use crate::serve::gpu::QuantWeightInfo;
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().unwrap();
        let cfg = config(1_024);
        let mut tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &cfg, 3);
        for off in [None, Some(""), Some("0")] {
            assert!(!resolve_staged_draft(off, &cfg, &tensors, 4).unwrap());
        }
        for lanes in 1..=4 {
            assert!(resolve_staged_draft(Some("1"), &cfg, &tensors, lanes).unwrap());
        }
        for invalid in ["true", "on", " 1", "2"] {
            let error = resolve_staged_draft(Some(invalid), &cfg, &tensors, 4).unwrap_err();
            assert!(error.to_string().contains("must be 0 or 1"), "{error}");
        }

        // Rows the staged tiles cannot cover fail at load, not at the first
        // encode.
        let short_block = DFlash2Config {
            block_size: 4,
            ..cfg.clone()
        };
        assert!(resolve_staged_draft(Some("1"), &short_block, &tensors, 2).unwrap());
        let error = resolve_staged_draft(Some("1"), &short_block, &tensors, 3).unwrap_err();
        assert!(
            error.to_string().contains("multiple of 8 draft rows"),
            "{error}"
        );

        let narrow = config(512);
        let narrow_tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &narrow, 3);
        let error = resolve_staged_draft(Some("1"), &narrow, &narrow_tensors, 4).unwrap_err();
        assert!(error.to_string().contains("hidden size"), "{error}");
        assert!(!resolve_staged_draft(Some("0"), &narrow, &narrow_tensors, 4).unwrap());

        tensors.layers[0].gate_up_halves = None;
        let error = resolve_staged_draft(Some("1"), &cfg, &tensors, 4).unwrap_err();
        assert!(error.to_string().contains("64-row tiles"), "{error}");

        tensors.context_projection = DFlash2Projection::Dense {
            weight: device.alloc_buffer(2, DType::BF16, vec![1, 1]).unwrap(),
        };
        let error = resolve_staged_draft(Some("1"), &cfg, &tensors, 4).unwrap_err();
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

    struct Fixture {
        device: MlxDevice,
        registry: KernelRegistry,
        cfg: DFlash2Config,
        tensors: DFlash2ModelTensors,
        workspace: DFlash2Workspace,
    }

    /// A synthetic checkpoint whose rings are filled past one wrap and whose
    /// draft input rows are set.
    fn fixture(lanes: usize, staged: bool) -> Fixture {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let cfg = config(1_024);
        let tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &cfg, 5);
        assert!(resolve_staged_draft(Some("1"), &cfg, &tensors, lanes).unwrap());
        let mut state = 17_u64;
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
        Fixture {
            device,
            registry,
            cfg,
            tensors,
            workspace,
        }
    }

    /// One draft forward of [`fixture`] through a dense head.
    fn run(lanes: usize, staged: bool) -> Outputs {
        let Fixture {
            device,
            mut registry,
            cfg,
            tensors,
            mut workspace,
        } = fixture(lanes, staged);
        let mut head = device
            .alloc_buffer(
                cfg.vocab_size * cfg.hidden_size * 2,
                DType::BF16,
                vec![cfg.vocab_size, cfg.hidden_size],
            )
            .unwrap();
        let mut state = 29_u64;
        for weight in head.as_mut_slice::<bf16>().unwrap() {
            *weight = bf16::from_f32(0.05 * random(&mut state));
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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

    /// A Q4_K head of `cfg`'s shape with small finite weights.
    fn q4k_head(device: &MlxDevice, cfg: &DFlash2Config) -> MlxQWeight {
        const BLOCK_BYTES: usize = 144;
        let bytes = cfg.vocab_size * cfg.hidden_size / 256 * BLOCK_BYTES;
        let mut buffer = device.alloc_buffer(bytes, DType::U8, vec![bytes]).unwrap();
        for (block, bytes) in buffer
            .as_mut_slice::<u8>()
            .unwrap()
            .chunks_exact_mut(BLOCK_BYTES)
            .enumerate()
        {
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = ((block * 31 + index * 13 + 7) & 255) as u8;
            }
            bytes[0..2].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
            bytes[2..4].copy_from_slice(&half::f16::from_f32(0.0078125).to_le_bytes());
            for scale in &mut bytes[4..16] {
                *scale &= 0x3f;
            }
        }
        MlxQWeight {
            buffer,
            info: QuantWeightInfo {
                ggml_dtype: GgmlType::Q4_K,
                rows: cfg.vocab_size,
                cols: cfg.hidden_size,
            },
            affine: None,
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        }
    }

    /// Encode one draft forward through a native head; commit it or abandon
    /// it as a failed encode would.
    fn forward_native(fixture: &mut Fixture, head: &MlxQWeight, commit: bool) -> Vec<f32> {
        forward_head(fixture, DFlash2VocabularyHead::Native(head), commit)
    }

    fn forward_head(
        fixture: &mut Fixture,
        head: DFlash2VocabularyHead<'_>,
        commit: bool,
    ) -> Vec<f32> {
        // A forward overwrites its input rows; every call starts from the same.
        let mut state = 43_u64;
        for value in fixture.workspace.hidden[0].as_mut_slice::<f32>().unwrap() {
            *value = 2.0 * random(&mut state);
        }
        let mut encoder = fixture.device.command_encoder().unwrap();
        dispatch_dflash2_forward(
            &mut encoder,
            &mut fixture.registry,
            &fixture.device,
            &fixture.cfg,
            &fixture.tensors,
            head,
            &mut fixture.workspace,
        )
        .unwrap();
        if commit {
            encoder.commit_and_wait().unwrap();
            if let Some(staged) = fixture.workspace.staged.as_mut() {
                staged.command_committed();
            }
        }
        fixture.workspace.logits.as_slice::<f32>().unwrap().to_vec()
    }

    /// The target's shared planes give the logits of the draft's own copy,
    /// bit for bit, and the draft then prepares no head of its own.
    #[test]
    fn shared_prepared_head_gives_the_bits_of_the_drafts_own_copy() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut own = fixture(4, true);
        let head = q4k_head(&own.device, &own.cfg);
        let own_logits = forward_native(&mut own, &head, true);
        assert!(own.workspace.staged.as_ref().unwrap().head.is_some());
        assert!(own_logits.iter().any(|value| *value != 0.0));

        let mut shared = fixture(4, true);
        let head = q4k_head(&shared.device, &shared.cfg);
        let params = head.matmul_params(32).unwrap();
        let mut encoder = shared.device.command_encoder().unwrap();
        let prepared = prepare_repacked_gguf_weight(
            &mut encoder,
            &mut shared.registry,
            &shared.device,
            &head.buffer,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        for _ in 0..2 {
            let logits = forward_head(
                &mut shared,
                DFlash2VocabularyHead::SharedPrepared {
                    weight: &head,
                    prepared: &prepared,
                },
                true,
            );
            assert!(
                logits
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(own_logits.iter().map(|v| v.to_bits())),
                "shared planes changed the draft logits"
            );
        }
        let staged = shared.workspace.staged.as_ref().unwrap();
        assert!(
            staged.head.is_none(),
            "the draft prepared a head beside the shared one"
        );
    }

    /// A target that counts how often it is asked for its shared head.
    struct CountingTarget<'a> {
        head: &'a MlxQWeight,
        prepared: &'a PreparedGgufWeight,
        asked: std::cell::Cell<usize>,
    }

    impl DFlash2TargetFamily for CountingTarget<'_> {
        fn family_name(&self) -> &'static str {
            "counting"
        }

        fn target_shape(&self) -> DFlash2TargetShape<'_> {
            DFlash2TargetShape {
                architecture: "counting",
                hidden_size: self.head.info.cols,
                vocab_size: self.head.info.rows,
                num_hidden_layers: 1,
            }
        }

        fn embedding(&self) -> Result<&MlxQWeight> {
            Ok(self.head)
        }

        fn vocabulary_head(&self) -> Result<&MlxQWeight> {
            Ok(self.head)
        }

        fn shared_prepared_vocabulary_head(
            &self,
            _registry: &mut KernelRegistry,
            _device: &MlxDevice,
        ) -> Result<Option<&PreparedGgufWeight>> {
            self.asked.set(self.asked.get() + 1);
            Ok(Some(self.prepared))
        }

        fn embedding_scale(&self) -> f32 {
            1.0
        }
    }

    /// The target is asked for its shared head (which prepares it) only by a
    /// command that reads prepared planes: the staged route at 32 draft rows.
    #[test]
    fn the_shared_head_is_requested_only_by_a_command_that_reads_it() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // (lanes, staged, reads the prepared head)
        for (lanes, staged, reads) in [(4, true, true), (4, false, false), (2, true, false)] {
            let mut fixture = fixture(lanes, staged);
            assert_eq!(
                reads_prepared_head(&fixture.workspace, &fixture.cfg),
                reads,
                "{lanes} lanes, staged {staged}"
            );
            let head = q4k_head(&fixture.device, &fixture.cfg);
            let params = head.matmul_params(32).unwrap();
            let mut encoder = fixture.device.command_encoder().unwrap();
            let prepared = prepare_repacked_gguf_weight(
                &mut encoder,
                &mut fixture.registry,
                &fixture.device,
                &head.buffer,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            let target = CountingTarget {
                head: &head,
                prepared: &prepared,
                asked: std::cell::Cell::new(0),
            };
            let vocabulary_head = target_vocabulary_head(
                &target,
                &mut fixture.registry,
                &fixture.device,
                reads_prepared_head(&fixture.workspace, &fixture.cfg),
            )
            .unwrap();
            assert_eq!(
                matches!(
                    vocabulary_head,
                    DFlash2VocabularyHead::SharedPrepared { .. }
                ),
                reads,
                "{lanes} lanes, staged {staged}"
            );
            assert_eq!(target.asked.get(), usize::from(reads));
            // The command then runs the head it was handed.
            let logits = forward_head(&mut fixture, vocabulary_head, true);
            assert!(logits.iter().any(|value| *value != 0.0));
            let own = fixture
                .workspace
                .staged
                .as_ref()
                .is_some_and(|staged| staged.head.is_some());
            assert!(!own, "the draft prepared a head of its own");
        }
    }

    #[test]
    fn prepared_head_is_recorded_only_after_its_command_commits() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let prepared =
            |fixture: &Fixture| fixture.workspace.staged.as_ref().unwrap().head.is_some();
        let mut staged = fixture(4, true);
        let head = q4k_head(&staged.device, &staged.cfg);

        // A command abandoned after it encoded the repack never ran it.
        forward_native(&mut staged, &head, false);
        assert!(
            !prepared(&staged),
            "an uncommitted command recorded its head"
        );
        let logits = forward_native(&mut staged, &head, true);
        assert!(prepared(&staged));
        let again = forward_native(&mut staged, &head, true);
        assert!(
            logits
                .iter()
                .map(|v| v.to_bits())
                .eq(again.iter().map(|v| v.to_bits())),
            "the recorded head gives other logits than the command that prepared it"
        );

        // The prepared tile computes the stock head's product.
        let mut default = fixture(4, false);
        let stock = forward_native(&mut default, &head, true);
        let error = relative_rms(&logits, &stock);
        assert!(
            error <= 5.0e-3,
            "prepared head differs by relative RMS {error}"
        );

        // Fewer than four lanes keep the stock head.
        let mut narrow = fixture(3, true);
        forward_native(&mut narrow, &head, true);
        assert!(!prepared(&narrow));
    }

    /// Every ring's bits after appending `rows` capture rows to `lane` at
    /// `start`: by the decode-shaped commits of `block_size` rows on the
    /// lane's own row group, one command buffer each, or by lane passes.
    fn ingest_rings(
        lanes: usize,
        staged: bool,
        lane: usize,
        start: usize,
        rows: usize,
        lane_passes: bool,
    ) -> Vec<u16> {
        use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
        use crate::inference::spec_decode::dflash2::runtime::ingest_capture_lane_passes;

        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let cfg = config(1_024);
        let tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &cfg, 5);
        let mut workspace = DFlash2Workspace::new(&device, &cfg, lanes, staged).unwrap();
        assert_eq!(workspace.staged.is_some(), staged);
        let width = cfg.fc_input_dim();
        let mut state = 71_u64;
        let capture = DFlashCaptureSession {
            target_layer_ids: cfg.target_layer_ids.clone(),
            hidden_output: (0..rows * width)
                .map(|_| 2.0 * random(&mut state))
                .collect(),
            per_position_argmaxes: None,
            seq_len: rows,
            hidden_size: width / cfg.target_layer_ids.len(),
        };
        // The other lanes' rows hold whatever the last cycle left there.
        for value in workspace
            .captured_target_hidden
            .as_mut_slice::<f32>()
            .unwrap()
        {
            *value = 2.0 * random(&mut state);
        }
        if lane_passes {
            ingest_capture_lane_passes(
                &device,
                &mut registry,
                &cfg,
                &tensors,
                &workspace,
                &capture,
                lane,
                start,
                0..rows,
            )
            .unwrap();
        } else {
            let mut lengths: Vec<u32> = (0..lanes as u32).map(|other| 17 * other + 3).collect();
            lengths[lane] = start as u32;
            let mut retained = device
                .alloc_buffer(lanes * 4, DType::U32, vec![lanes])
                .unwrap();
            let lane_rows = lane * cfg.block_size * width..(lane + 1) * cfg.block_size * width;
            for chunk in capture.hidden_output.chunks(cfg.block_size * width) {
                let destination = &mut workspace
                    .captured_target_hidden
                    .as_mut_slice::<f32>()
                    .unwrap()[lane_rows.clone()];
                destination.fill(0.0);
                destination[..chunk.len()].copy_from_slice(chunk);
                prepare_decode_positions(&mut workspace, &cfg, &lengths).unwrap();
                let counts = retained.as_mut_slice::<u32>().unwrap();
                counts.fill(0);
                counts[lane] = (chunk.len() / width) as u32;
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
                lengths[lane] += (chunk.len() / width) as u32;
            }
        }
        workspace
            .ring_keys
            .iter()
            .chain(&workspace.ring_values)
            .flat_map(|ring| ring.as_slice::<bf16>().unwrap())
            .map(|value| value.to_bits())
            .collect()
    }

    /// A prefill capture appended in passes of the workspace's whole row
    /// capacity leaves, bit for bit, the rings that decode-shaped commits of
    /// `block_size` rows leave: on both draft routes, at every lane count,
    /// for a partial block, a chunk of several passes with a partial last
    /// group, a chunk that wraps the ring, and one longer than the ring.
    #[test]
    fn lane_passes_write_the_rings_of_block_commits() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for staged in [false, true] {
            for lanes in [1, 3, 4] {
                for (lane, start, rows) in [
                    (0, 0, 5),
                    (lanes - 1, 3, 61),
                    (0, 250, 200),
                    (lanes - 1, 10, 300),
                ] {
                    let blocks = ingest_rings(lanes, staged, lane, start, rows, false);
                    let passes = ingest_rings(lanes, staged, lane, start, rows, true);
                    assert!(blocks.iter().any(|&bits| bits != 0), "nothing was appended");
                    let differing = blocks
                        .iter()
                        .zip(&passes)
                        .filter(|(block, pass)| block != pass)
                        .count();
                    assert!(
                        differing == 0,
                        "staged {staged}, {lanes} lanes, lane {lane}, {rows} rows at {start}: \
                         {differing} of {} ring values differ",
                        blocks.len()
                    );
                }
            }
        }
    }

    /// Draft context routes of the ingest gate below.
    #[derive(Clone, Copy, Debug)]
    enum IngestRoute {
        /// Dense BF16 context projections (the default draft projection).
        Dense,
        /// AF4-G64 projections in one command.
        Af4,
        /// AF4-G64 projections on the per-product staged draft workspace.
        Af4Staged,
    }

    fn dense_bf16_projection(
        device: &MlxDevice,
        shape: &[usize],
        state: &mut u64,
        scale: f32,
    ) -> DFlash2Projection {
        let elements: usize = shape.iter().product();
        let mut weight = device
            .alloc_buffer(elements * 2, DType::BF16, shape.to_vec())
            .unwrap();
        for bits in weight.as_mut_slice::<u16>().unwrap() {
            *bits = bf16::from_f32(scale * random(state)).to_bits();
        }
        DFlash2Projection::Dense { weight }
    }

    /// Rings after every step of a multi-lane scenario with two target
    /// layers: chunks of 5 to 300 rows on two lanes, a lane reset between
    /// them, a ring wrap and a chunk longer than the ring. Arm 0: the
    /// decode-shaped block commits the ingest replaced, the capture
    /// converted from layer-major to row-major by `extract_drafter_concat`;
    /// arm 1: lane passes; arm 2: lane passes with every chunk split into
    /// two row ranges (a non-zero row start, as the cue fold appends);
    /// arm 3: lane passes fed the row-major layout (negative control).
    fn ingest_scenario(route: IngestRoute, lanes: usize, arm: usize) -> Vec<Vec<u16>> {
        use crate::inference::spec_decode::dflash::hidden_capture::{
            extract_drafter_concat, DFlashCaptureSession,
        };
        use crate::inference::spec_decode::dflash2::runtime::ingest_capture_lane_passes;

        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let cfg = DFlash2Config {
            num_target_layers: 2,
            target_layer_ids: vec![3, 9],
            ..config(1_024)
        };
        let mut tensors = DFlash2ModelTensors::synthetic_af4_g64(&device, &cfg, 5);
        let staged = matches!(route, IngestRoute::Af4Staged);
        if matches!(route, IngestRoute::Dense) {
            let mut state = 991_u64;
            let kv = cfg.num_key_value_heads * cfg.head_dim;
            tensors.context_projection = dense_bf16_projection(
                &device,
                &[cfg.hidden_size, cfg.fc_input_dim()],
                &mut state,
                0.05,
            );
            for layer in &mut tensors.layers {
                layer.context_kv_projection =
                    dense_bf16_projection(&device, &[2 * kv, cfg.hidden_size], &mut state, 0.05);
            }
        }
        let mut workspace = DFlash2Workspace::new(&device, &cfg, lanes, staged).unwrap();
        assert_eq!(workspace.staged.is_some(), staged);
        let other = lanes - 1;
        let ops: [Option<(usize, usize)>; 8] = [
            Some((0, 5)),
            Some((other, 61)),
            Some((0, 27)),
            Some((other, 300)),
            None,
            Some((0, 13)),
            Some((other, 9)),
            Some((0, 250)),
        ];
        let width = cfg.fc_input_dim();
        let layers = cfg.target_layer_ids.len();
        let hidden = width / layers;
        let mut lengths = vec![0_u32; lanes];
        let mut state = 4_242_u64;
        for value in workspace
            .captured_target_hidden
            .as_mut_slice::<f32>()
            .unwrap()
        {
            *value = 2.0 * random(&mut state);
        }
        let mut data_state = 77_u64;
        let mut retained = device
            .alloc_buffer(lanes * 4, DType::U32, vec![lanes])
            .unwrap();
        let mut snapshots = Vec::new();
        for op in ops {
            match op {
                None => {
                    // Lane 0 starts over: its ring is cleared and refilled.
                    let lane_elements = cfg.num_key_value_heads * cfg.sliding_window * cfg.head_dim;
                    for ring in workspace
                        .ring_keys
                        .iter_mut()
                        .chain(workspace.ring_values.iter_mut())
                    {
                        ring.as_mut_slice::<u16>().unwrap()[..lane_elements].fill(0);
                    }
                    lengths[0] = 0;
                }
                Some((lane, rows)) => {
                    let capture = DFlashCaptureSession {
                        target_layer_ids: cfg.target_layer_ids.clone(),
                        hidden_output: (0..layers * rows * hidden)
                            .map(|_| 2.0 * random(&mut data_state))
                            .collect(),
                        per_position_argmaxes: None,
                        seq_len: rows,
                        hidden_size: hidden,
                    };
                    let start = lengths[lane] as usize;
                    let concatenated = extract_drafter_concat(
                        &capture.hidden_output,
                        &capture.target_layer_ids,
                        &cfg.target_layer_ids,
                        rows,
                        hidden,
                    )
                    .unwrap();
                    let mut lane_passes =
                        |capture: &DFlashCaptureSession,
                         start: usize,
                         range: std::ops::Range<usize>| {
                            ingest_capture_lane_passes(
                                &device,
                                &mut registry,
                                &cfg,
                                &tensors,
                                &workspace,
                                capture,
                                lane,
                                start,
                                range,
                            )
                            .unwrap();
                        };
                    match arm {
                        0 => {
                            let lane_rows =
                                lane * cfg.block_size * width..(lane + 1) * cfg.block_size * width;
                            for chunk in concatenated.chunks(cfg.block_size * width) {
                                let destination = &mut workspace
                                    .captured_target_hidden
                                    .as_mut_slice::<f32>()
                                    .unwrap()[lane_rows.clone()];
                                destination.fill(0.0);
                                destination[..chunk.len()].copy_from_slice(chunk);
                                prepare_decode_positions(&mut workspace, &cfg, &lengths).unwrap();
                                let counts = retained.as_mut_slice::<u32>().unwrap();
                                counts.fill(0);
                                counts[lane] = (chunk.len() / width) as u32;
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
                                lengths[lane] += (chunk.len() / width) as u32;
                            }
                        }
                        1 => {
                            lane_passes(&capture, start, 0..rows);
                            lengths[lane] += rows as u32;
                        }
                        2 => {
                            let split = rows / 3;
                            for range in [0..split, split..rows] {
                                lane_passes(&capture, start + range.start, range);
                            }
                            lengths[lane] += rows as u32;
                        }
                        _ => {
                            let row_major = DFlashCaptureSession {
                                target_layer_ids: cfg.target_layer_ids.clone(),
                                hidden_output: concatenated.clone(),
                                per_position_argmaxes: None,
                                seq_len: rows,
                                hidden_size: hidden,
                            };
                            lane_passes(&row_major, start, 0..rows);
                            lengths[lane] += rows as u32;
                        }
                    }
                }
            }
            snapshots.push(
                workspace
                    .ring_keys
                    .iter()
                    .chain(&workspace.ring_values)
                    .flat_map(|ring| ring.as_slice::<bf16>().unwrap())
                    .map(|value| value.to_bits())
                    .collect(),
            );
        }
        snapshots
    }

    /// The ingest gate on what the one-layer AF4 gate above cannot see: the
    /// capture's layer-major to row-major conversion (two target layers),
    /// the Dense BF16 draft projection, a row range that does not start at
    /// row 0, a lane reset between chunks and two lanes interleaved.
    /// (Adopted from the independent review of the change.)
    #[test]
    fn lane_passes_equal_block_commits_with_two_layers_on_dense_and_af4_routes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for route in [IngestRoute::Dense, IngestRoute::Af4, IngestRoute::Af4Staged] {
            for lanes in 1..=4_usize {
                let blocks = ingest_scenario(route, lanes, 0);
                let passes = ingest_scenario(route, lanes, 1);
                let ranges = ingest_scenario(route, lanes, 2);
                let mut failed = Vec::new();
                for (step, block) in blocks.iter().enumerate() {
                    let nonzero = block.iter().filter(|&&bits| bits != 0).count();
                    let differing =
                        |other: &[u16]| block.iter().zip(other).filter(|(a, b)| a != b).count();
                    // Step 4 is the lane reset: with one lane it empties the rings.
                    if (nonzero == 0 && !(step == 4 && lanes == 1))
                        || differing(&passes[step]) != 0
                        || differing(&ranges[step]) != 0
                    {
                        failed.push(step);
                    }
                }
                assert!(
                    failed.is_empty(),
                    "{route:?} lanes {lanes}: steps {failed:?} differ from the block commits"
                );
            }
            // The comparison sees the capture layout: the row-major layout
            // fed as if it were layer-major leaves other rings.
            let blocks = ingest_scenario(route, 2, 0);
            let row_major = ingest_scenario(route, 2, 3);
            assert!(
                blocks[3] != row_major[3],
                "{route:?}: the capture layout does not reach the rings"
            );
        }
    }
}
