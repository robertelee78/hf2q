//! Verified checkpoint ownership and bounded DFlash2 cycle execution.

use std::path::Path;

use sha2::{Digest, Sha256};

#[cfg(test)]
use super::forward::dispatch_layer;
use super::{
    adapters::{dispatch_dflash2_input, target_vocabulary_head, DFlash2TargetFamily},
    config::DFlash2Config,
    forward::{
        dispatch_dflash2_context_commit, dispatch_dflash2_forward, prepare_decode_positions,
    },
    schema::{DFlash2CheckpointDescriptor, DFlash2FamilyAdapter},
    selector::dispatch_dflash2_selection,
    staged::{resolve_staged_draft, DFLASH2_DRAFT_STAGED_ENV},
    tensors::DFlash2ModelTensors,
    verify_budget::{DFlash2VerifyBudget, VerifyBudgetLane},
    weights::DFlash2WeightsFile,
    workspace::DFlash2Workspace,
};
use crate::inference::spec_decode::dflash::hidden_capture::{
    extract_drafter_concat, DFlashCaptureSession,
};
use anyhow::{bail, ensure, Context, Result};
use mlx_native::{
    ops::dflash2_selector::{
        dispatch_dflash2_accept_greedy, DFlash2GreedyAcceptanceParams, DFLASH2_MAX_LANES,
    },
    CommandEncoder, KernelRegistry, MlxBuffer, MlxDevice,
};

const GPU_TARGET_TOKENS_ENV: &str = "HF2Q_DFLASH2_GPU_TARGET_TOKENS";

fn resolve_gpu_target_tokens(
    descriptor: &DFlash2CheckpointDescriptor,
    raw: Option<&str>,
) -> Result<bool> {
    let qualified = descriptor.family_adapter == DFlash2FamilyAdapter::QwenDense;
    let supported = matches!(
        descriptor.family_adapter,
        DFlash2FamilyAdapter::QwenDense | DFlash2FamilyAdapter::QwenMoe
    );
    match raw {
        None => Ok(qualified),
        Some("0" | "off" | "false") => Ok(false),
        Some("1" | "on" | "true") if supported => Ok(true),
        Some("1" | "on" | "true") => bail!(
            "{GPU_TARGET_TOKENS_ENV}=1 is not qualified for DFlash2 adapter {:?}",
            descriptor.family_adapter
        ),
        Some(value) => {
            bail!("{GPU_TARGET_TOKENS_ENV} must be one of 0/off/false/1/on/true, got {value:?}")
        }
    }
}

struct DFlash2LaneBinding {
    hasher: Sha256,
    tokens: usize,
    rewind_tokens: Option<Vec<u32>>,
}

impl DFlash2LaneBinding {
    fn new(rewind_limit: usize) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"hf2q-dflash2-lane-v1");
        Self {
            hasher,
            tokens: 0,
            rewind_tokens: Some(Vec::with_capacity(rewind_limit)),
        }
    }

    fn append(&mut self, tokens: &[u32], rewind_limit: usize) {
        for token in tokens {
            self.hasher.update(token.to_le_bytes());
        }
        self.tokens += tokens.len();
        if let Some(prefix) = self.rewind_tokens.as_mut() {
            if self.tokens <= rewind_limit {
                prefix.extend_from_slice(tokens);
            } else {
                self.rewind_tokens = None;
            }
        }
    }

    fn matches(&self, tokens: &[u32]) -> bool {
        if self.tokens != tokens.len() {
            return false;
        }
        let mut hasher = Sha256::new();
        hasher.update(b"hf2q-dflash2-lane-v1");
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        self.hasher.clone().finalize() == hasher.finalize()
    }

    fn rewind(&mut self, tokens: &[u32], rewind_limit: usize) -> bool {
        if self.matches(tokens) {
            return true;
        }
        let Some(prefix) = self.rewind_tokens.as_ref() else {
            return false;
        };
        if tokens.len() >= self.tokens || prefix.get(..tokens.len()) != Some(tokens) {
            return false;
        }
        *self = Self::new(rewind_limit);
        self.append(tokens, rewind_limit);
        true
    }
}

/// GPU-resident DFlash2 checkpoint and persistent per-slot draft rings.
pub struct DFlash2Runtime {
    pub descriptor: &'static DFlash2CheckpointDescriptor,
    pub config: DFlash2Config,
    pub tensors: DFlash2ModelTensors,
    pub workspace: DFlash2Workspace,
    lane_lengths: Vec<usize>,
    lane_bindings: Vec<DFlash2LaneBinding>,
    verify_budget: DFlash2VerifyBudget,
    gpu_target_tokens: bool,
}

impl DFlash2Runtime {
    /// Verify the immutable checkpoint identity before uploading any tensor.
    pub fn load_verified<T: DFlash2TargetFamily>(
        device: &MlxDevice,
        descriptor: &'static DFlash2CheckpointDescriptor,
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        target: &T,
        lanes: usize,
    ) -> Result<Self> {
        let config_bytes = std::fs::read(config_path.as_ref())
            .with_context(|| format!("read DFlash2 config {}", config_path.as_ref().display()))?;
        let config = descriptor
            .parse_config(&config_bytes)
            .context("validate DFlash2 config identity")?;
        target
            .validate(&config)
            .context("bind DFlash2 checkpoint to target")?;
        let weights_file = DFlash2WeightsFile::open_verified(weights_path.as_ref(), descriptor)
            .with_context(|| {
                format!("verify DFlash2 weights {}", weights_path.as_ref().display())
            })?;
        let weights = weights_file
            .load_verified(&config, descriptor)
            .context("validate DFlash2 tensor manifest")?;
        let tensors = DFlash2ModelTensors::upload(device, &config, &weights)
            .context("upload DFlash2 tensors")?;
        let staged = resolve_staged_draft(
            std::env::var(DFLASH2_DRAFT_STAGED_ENV).ok().as_deref(),
            &config,
            &tensors,
        )?;
        let workspace = DFlash2Workspace::new(device, &config, lanes, staged)
            .context("allocate DFlash2 workspace")?;
        let gpu_target_tokens = resolve_gpu_target_tokens(
            descriptor,
            std::env::var(GPU_TARGET_TOKENS_ENV).ok().as_deref(),
        )?;
        tracing::info!(
            checkpoint = descriptor.id,
            gpu_target_tokens,
            staged_draft = staged,
            "DFlash2 transaction policy resolved"
        );
        let rewind_limit = config.sliding_window;
        Ok(Self {
            descriptor,
            config,
            tensors,
            workspace,
            lane_lengths: vec![0; lanes],
            lane_bindings: (0..lanes)
                .map(|_| DFlash2LaneBinding::new(rewind_limit))
                .collect(),
            verify_budget: DFlash2VerifyBudget::default(),
            gpu_target_tokens,
        })
    }

    pub(crate) fn gpu_target_tokens(&self) -> bool {
        self.gpu_target_tokens
    }

    /// Execute one physical anchor-plus-seven-proposal graph for every slot
    /// and wait until host proposal readback is safe.
    #[allow(clippy::too_many_arguments)]
    pub fn run_draft<T: DFlash2TargetFamily>(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        target: &T,
        anchors: &[u32],
        logical_positions: &[u32],
        temperatures: &[f32],
        sampling_mask: u32,
        maximum_verify_rows: &[u32],
        uniforms: &[f32],
    ) -> Result<()> {
        let target_lane_order = (0..self.workspace.max_lanes as u32).collect::<Vec<_>>();
        self.run_draft_inner(
            device,
            registry,
            target,
            anchors,
            logical_positions,
            temperatures,
            sampling_mask,
            maximum_verify_rows,
            &target_lane_order,
            uniforms,
            true,
        )
    }

    /// Submit the draft graph without a host wait. The graph also composes
    /// `workspace.target_tokens`; a same-queue target embedding may consume it
    /// immediately, while host proposal access remains invalid until that
    /// target command reaches its completion boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_draft_for_gpu_target<T: DFlash2TargetFamily>(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        target: &T,
        anchors: &[u32],
        logical_positions: &[u32],
        temperatures: &[f32],
        sampling_mask: u32,
        maximum_verify_rows: &[u32],
        target_lane_order: &[u32],
        uniforms: &[f32],
    ) -> Result<()> {
        self.run_draft_inner(
            device,
            registry,
            target,
            anchors,
            logical_positions,
            temperatures,
            sampling_mask,
            maximum_verify_rows,
            target_lane_order,
            uniforms,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_draft_inner<T: DFlash2TargetFamily>(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        target: &T,
        anchors: &[u32],
        logical_positions: &[u32],
        temperatures: &[f32],
        sampling_mask: u32,
        maximum_verify_rows: &[u32],
        target_lane_order: &[u32],
        uniforms: &[f32],
        wait: bool,
    ) -> Result<()> {
        let lanes = self.workspace.max_lanes;
        let proposals = self.config.proposal_tokens();
        ensure!(
            uniforms.len() == lanes * proposals,
            "DFlash2 uniform count {} != lanes {lanes} * proposals {proposals}",
            uniforms.len()
        );
        ensure!(
            logical_positions
                .iter()
                .zip(&self.lane_lengths)
                .all(|(position, length)| usize::try_from(*position) == Ok(*length)),
            "DFlash2 draft positions {logical_positions:?} do not match lane contexts {:?}",
            self.lane_lengths
        );
        self.workspace
            .uniforms
            .as_mut_slice::<f32>()?
            .copy_from_slice(uniforms);
        prepare_decode_positions(&mut self.workspace, &self.config, logical_positions)?;
        let mut encoder = device
            .command_encoder()
            .context("create DFlash2 draft encoder")?;
        dispatch_dflash2_input(
            &self.tensors,
            &mut encoder,
            registry,
            device,
            &self.config,
            target,
            anchors,
            &mut self.workspace,
        )?;
        let vocabulary_head = target_vocabulary_head(target)?;
        dispatch_dflash2_forward(
            &mut encoder,
            registry,
            device,
            &self.config,
            &self.tensors,
            vocabulary_head,
            &mut self.workspace,
        )?;
        let verify_model = if !wait && target.adaptive_verify_budget() {
            Some(self.verify_budget.model(target_lane_order.len())?)
        } else {
            None
        };
        dispatch_dflash2_selection(
            &mut encoder,
            registry,
            device,
            &self.config,
            &self.tensors,
            &self.workspace,
            anchors,
            temperatures,
            sampling_mask,
            maximum_verify_rows,
            target_lane_order,
            verify_model,
        )?;
        if wait {
            encoder
                .commit_and_wait_labeled("adr067.dflash2_draft")
                .context("execute DFlash2 draft cycle")?;
        } else {
            encoder.commit_labeled("adr067.dflash2_draft");
        }
        Ok(())
    }

    pub fn proposals(&self) -> Result<&[u32]> {
        Ok(self.workspace.proposed_tokens.as_slice::<u32>()?)
    }

    pub fn verify_rows(&self) -> Result<&[u32]> {
        Ok(self.workspace.verify_rows.as_slice::<u32>()?)
    }

    pub(crate) fn select_verify_rows(
        &mut self,
        lane_indices: &[usize],
        maximum_rows: &[u32],
    ) -> Result<()> {
        let rows = self
            .verify_budget
            .select_rows(&self.workspace, lane_indices, maximum_rows)?;
        let selected = self.workspace.verify_rows.as_mut_slice::<u32>()?;
        for (&lane, rows) in lane_indices.iter().zip(rows) {
            selected[lane] = u32::try_from(rows)?;
        }
        Ok(())
    }

    pub(crate) fn observe_verify_budget(
        &mut self,
        lanes: &[VerifyBudgetLane],
        milliseconds: f64,
    ) -> Result<()> {
        self.verify_budget
            .observe(&self.workspace, lanes, milliseconds)
    }
    pub fn lane_lengths(&self) -> &[usize] {
        &self.lane_lengths
    }

    /// Align a resident draft ring with the exact target token prefix restored
    /// into this slot. Rewind is safe only before the first ring wrap; after
    /// that, overwritten older rows make an earlier prefix unrecoverable.
    pub fn align_lane_prefix(&mut self, lane: usize, tokens: &[u32]) -> bool {
        let Some(length) = self.lane_lengths.get_mut(lane) else {
            return false;
        };
        let Some(binding) = self.lane_bindings.get_mut(lane) else {
            return false;
        };
        if binding.tokens != *length || !binding.rewind(tokens, self.config.sliding_window) {
            return false;
        }
        *length = tokens.len();
        true
    }

    /// Bind a successfully committed hidden-state suffix to its exact target
    /// token IDs. The ring and token identity must advance in lockstep.
    pub fn record_lane_tokens(&mut self, lane: usize, tokens: &[u32]) -> Result<()> {
        let lane_length = *self
            .lane_lengths
            .get(lane)
            .context("DFlash2 token-binding lane is out of range")?;
        let binding = self
            .lane_bindings
            .get_mut(lane)
            .context("DFlash2 token-binding lane is out of range")?;
        let bound_length = binding
            .tokens
            .checked_add(tokens.len())
            .context("DFlash2 token-binding length overflow")?;
        ensure!(
            bound_length == lane_length,
            "DFlash2 lane {lane} token binding {} + {} != ring length {lane_length}",
            binding.tokens,
            tokens.len()
        );
        binding.append(tokens, self.config.sliding_window);
        Ok(())
    }

    /// Append a target prefill capture to one physical draft-ring lane.
    ///
    /// The legacy capture is CPU layer-major. This method performs the one
    /// layout conversion, then projects and commits bounded eight-row chunks
    /// through the same transaction used after target verification.
    pub fn ingest_target_capture(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        capture: &DFlashCaptureSession,
        lane: usize,
        logical_start: usize,
    ) -> Result<()> {
        ensure!(
            lane < self.workspace.max_lanes,
            "DFlash2 capture lane {lane} is out of range"
        );
        ensure!(
            capture.target_layer_ids == self.config.target_layer_ids,
            "DFlash2 capture layers {:?} != checkpoint layers {:?}",
            capture.target_layer_ids,
            self.config.target_layer_ids
        );
        ensure!(
            logical_start == self.lane_lengths[lane],
            "DFlash2 capture starts at {logical_start}, lane {lane} is at {}",
            self.lane_lengths[lane]
        );
        let concatenated = extract_drafter_concat(
            &capture.hidden_output,
            &capture.target_layer_ids,
            &self.config.target_layer_ids,
            capture.seq_len,
            capture.hidden_size,
        )
        .context("convert target capture to DFlash2 concat layout")?;
        let feature_width = self.config.fc_input_dim();
        ensure!(
            concatenated.len() == capture.seq_len * feature_width,
            "DFlash2 concatenated capture has {} values, expected {}",
            concatenated.len(),
            capture.seq_len * feature_width
        );
        for (chunk_index, chunk) in concatenated
            .chunks(self.config.block_size * feature_width)
            .enumerate()
        {
            let rows = chunk.len() / feature_width;
            let lane_start = lane * self.config.block_size * feature_width;
            let destination = self
                .workspace
                .captured_target_hidden
                .as_mut_slice::<f32>()?;
            destination[lane_start..lane_start + self.config.block_size * feature_width].fill(0.0);
            destination[lane_start..lane_start + chunk.len()].copy_from_slice(chunk);

            let positions = self
                .lane_lengths
                .iter()
                .map(|length| {
                    u32::try_from(*length).context("DFlash2 context position exceeds u32")
                })
                .collect::<Result<Vec<_>>>()?;
            prepare_decode_positions(&mut self.workspace, &self.config, &positions)
                .with_context(|| format!("prepare DFlash2 capture chunk {chunk_index}"))?;
            let mut retained = vec![0_u32; self.workspace.max_lanes];
            retained[lane] = u32::try_from(rows)?;
            self.commit_target_context(device, registry, &retained)
                .with_context(|| format!("commit DFlash2 capture chunk {chunk_index}"))?;
        }
        Ok(())
    }

    /// Project retained target hidden rows and append only those rows to each
    /// physical slot's draft rings.
    pub fn commit_target_context(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        retained_counts: &[u32],
    ) -> Result<()> {
        ensure!(
            retained_counts.len() == self.workspace.max_lanes,
            "DFlash2 retained-count length {} != workspace lanes {}",
            retained_counts.len(),
            self.workspace.max_lanes
        );
        ensure!(
            retained_counts
                .iter()
                .all(|count| (*count as usize) <= self.config.block_size),
            "DFlash2 retained counts {retained_counts:?} exceed block size {}",
            self.config.block_size
        );
        self.workspace
            .retained_counts
            .as_mut_slice::<u32>()?
            .copy_from_slice(retained_counts);
        let mut encoder = device
            .command_encoder()
            .context("create DFlash2 context encoder")?;
        dispatch_dflash2_context_commit(
            &mut encoder,
            registry,
            device,
            &self.config,
            &self.tensors,
            &self.workspace.retained_counts,
            &self.workspace,
        )?;
        encoder
            .commit_and_wait()
            .context("commit DFlash2 target context")?;
        for (length, retained) in self.lane_lengths.iter_mut().zip(retained_counts) {
            *length = length
                .checked_add(*retained as usize)
                .context("DFlash2 lane length overflow")?;
        }
        Ok(())
    }

    /// Append greedy acceptance and draft-context settlement to the target
    /// output-head command. The caller owns the command's completion boundary
    /// and must validate the GPU counts before committing host state.
    pub(crate) fn encode_gpu_acceptance_and_context(
        &self,
        encoder: &mut CommandEncoder,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        target_tokens: &MlxBuffer,
        target_lane_order: &[u32],
        maximum_rows: &[u32],
        eos_tokens: &[u32],
    ) -> Result<()> {
        ensure!(
            !target_lane_order.is_empty()
                && target_lane_order.len() <= DFLASH2_MAX_LANES
                && maximum_rows.len() == self.workspace.max_lanes,
            "invalid DFlash2 GPU acceptance lane geometry"
        );
        ensure!(
            !eos_tokens.is_empty() && eos_tokens.len() <= 8,
            "DFlash2 GPU acceptance requires 1..=8 EOS tokens"
        );
        let mut physical_lane = [0_u32; DFLASH2_MAX_LANES];
        let mut maximum = [0_u32; DFLASH2_MAX_LANES];
        for (logical, &physical) in target_lane_order.iter().enumerate() {
            let physical_index = physical as usize;
            ensure!(
                physical_index < self.workspace.max_lanes
                    && !physical_lane[..logical].contains(&physical),
                "invalid DFlash2 GPU acceptance physical lane {physical}"
            );
            physical_lane[logical] = physical;
            maximum[logical] = maximum_rows[physical_index];
        }
        let mut eos_token = [0_u32; 8];
        eos_token[..eos_tokens.len()].copy_from_slice(eos_tokens);
        dispatch_dflash2_accept_greedy(
            encoder,
            registry,
            device,
            &self.workspace.proposed_tokens,
            target_tokens,
            &self.workspace.verify_rows,
            &self.workspace.retained_counts,
            &self.workspace.accepted_counts,
            DFlash2GreedyAcceptanceParams {
                physical_lane,
                maximum,
                eos_token,
                physical_lanes: self.workspace.max_lanes as u32,
                lanes: target_lane_order.len() as u32,
                rows_per_lane: self.config.block_size as u32,
                width: target_tokens.element_count() as u32,
                eos_count: eos_tokens.len() as u32,
            },
        )
        .context("encode DFlash2 greedy GPU acceptance")?;
        encoder.memory_barrier();
        dispatch_dflash2_context_commit(
            encoder,
            registry,
            device,
            &self.config,
            &self.tensors,
            &self.workspace.retained_counts,
            &self.workspace,
        )
        .context("encode DFlash2 GPU-resident target context commit")
    }

    pub(crate) fn gpu_acceptance_counts(&self) -> Result<(&[u32], &[u32])> {
        Ok((
            self.workspace.retained_counts.as_slice::<u32>()?,
            self.workspace.accepted_counts.as_slice::<u32>()?,
        ))
    }

    pub(crate) fn settle_gpu_context_lengths(&mut self, retained_counts: &[u32]) -> Result<()> {
        ensure!(
            retained_counts.len() == self.workspace.max_lanes
                && retained_counts
                    .iter()
                    .all(|count| (*count as usize) <= self.config.block_size),
            "invalid DFlash2 GPU retained counts {retained_counts:?}"
        );
        for (length, retained) in self.lane_lengths.iter_mut().zip(retained_counts) {
            *length = length
                .checked_add(*retained as usize)
                .context("DFlash2 lane length overflow")?;
        }
        Ok(())
    }

    /// Reset one scheduler slot before reassignment or incompatible prefix
    /// restore. Other slots remain byte-identical.
    pub fn clear_lane(&mut self, lane: usize) -> Result<()> {
        ensure!(
            lane < self.workspace.max_lanes,
            "DFlash2 lane {lane} is out of range"
        );
        let heads = self.config.num_key_value_heads;
        let width = self.config.sliding_window;
        let dim = self.config.head_dim;
        let lane_elements = heads * width * dim;
        for ring in self
            .workspace
            .ring_keys
            .iter_mut()
            .chain(self.workspace.ring_values.iter_mut())
        {
            let values = ring.as_mut_slice::<u16>()?;
            values[lane * lane_elements..(lane + 1) * lane_elements].fill(0);
        }
        self.lane_lengths[lane] = 0;
        self.lane_bindings[lane] = DFlash2LaneBinding::new(self.config.sliding_window);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::model::Qwen35Model;
    use crate::inference::spec_decode::dflash2::adapters::QwenDFlash2Adapter;
    use crate::inference::spec_decode::dflash2::schema::{QWEN36_DFLASH2, QWEN38_DFLASH2};
    use crate::serve::header::LoadProgress;
    use half::bf16;
    use safetensors::{tensor::TensorView, Dtype, SafeTensors};
    use std::path::PathBuf;

    #[test]
    fn lane_binding_rewinds_exact_unwrapped_prefix_only() {
        let mut binding = DFlash2LaneBinding::new(8);
        assert!(binding.matches(&[]));
        binding.append(&[7, 11, 13], 8);
        assert!(binding.matches(&[7, 11, 13]));
        assert!(!binding.matches(&[7, 11, 17]));
        assert!(binding.rewind(&[7, 11], 8));
        assert!(binding.matches(&[7, 11]));

        binding.append(&[13, 17], 3);
        assert!(!binding.rewind(&[7, 11, 13], 3));
    }

    #[test]
    fn gpu_target_token_policy_defaults_for_qualified_dense_qwen() {
        assert!(resolve_gpu_target_tokens(&QWEN38_DFLASH2, None).unwrap());
        assert!(!resolve_gpu_target_tokens(&QWEN36_DFLASH2, None).unwrap());
        assert!(!resolve_gpu_target_tokens(&QWEN38_DFLASH2, Some("0")).unwrap());
        assert!(resolve_gpu_target_tokens(&QWEN36_DFLASH2, Some("1")).unwrap());
        assert!(resolve_gpu_target_tokens(&QWEN38_DFLASH2, Some("invalid")).is_err());
    }

    fn float_values(view: TensorView<'_>) -> Vec<f32> {
        match view.dtype() {
            Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect(),
            Dtype::F32 => f32_values(view),
            dtype => panic!("unsupported fixture floating-point dtype {dtype:?}"),
        }
    }

    fn f32_values(view: TensorView<'_>) -> Vec<f32> {
        view.data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn i64_values(view: TensorView<'_>) -> Vec<i64> {
        view.data()
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    #[ignore = "requires HF2Q_QWEN36_GGUF, HF2Q_QWEN36_DFLASH2_CONFIG, HF2Q_QWEN36_DFLASH2_WEIGHTS and Apple Metal"]
    fn qwen36_verified_checkpoint_runs_one_real_draft_cycle() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let gguf_path =
            PathBuf::from(std::env::var_os("HF2Q_QWEN36_GGUF").expect("HF2Q_QWEN36_GGUF"));
        let config_path = PathBuf::from(
            std::env::var_os("HF2Q_QWEN36_DFLASH2_CONFIG").expect("HF2Q_QWEN36_DFLASH2_CONFIG"),
        );
        let weights_path = PathBuf::from(
            std::env::var_os("HF2Q_QWEN36_DFLASH2_WEIGHTS").expect("HF2Q_QWEN36_DFLASH2_WEIGHTS"),
        );
        let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open Qwen3.6 GGUF");
        let mut progress = LoadProgress::new(false, 1, 0);
        let model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load Qwen3.6");
        model
            .ensure_gpu_cache_primed()
            .expect("prime Qwen3.6 GPU cache");
        model
            .with_gpu_cache_mut(|device, registry| {
                let config_bytes = std::fs::read(&config_path)?;
                let config = QWEN36_DFLASH2.parse_config(&config_bytes)?;
                let adapter = QwenDFlash2Adapter::new(&model, &config)?;
                let mut runtime = DFlash2Runtime::load_verified(
                    device,
                    &QWEN36_DFLASH2,
                    &config_path,
                    &weights_path,
                    &adapter,
                    1,
                )?;
                runtime
                    .workspace
                    .captured_target_hidden
                    .as_mut_slice::<f32>()?
                    .fill(0.0);
                runtime.run_draft(
                    device,
                    registry,
                    &adapter,
                    &[1],
                    &[0],
                    &[0.0],
                    0,
                    &[8],
                    &[0.5; 7],
                )?;
                ensure!(
                    runtime
                        .proposals()?
                        .iter()
                        .all(|token| (*token as usize) < runtime.config.vocab_size),
                    "real DFlash2 proposal exceeds target vocabulary"
                );
                ensure!(
                    runtime.verify_rows()? == [8],
                    "real DFlash2 did not verify all rows"
                );
                let mut expected_target_tokens = vec![1_u32];
                expected_target_tokens.extend_from_slice(runtime.proposals()?);
                ensure!(
                    runtime.workspace.target_tokens.as_slice::<u32>()?
                        [..expected_target_tokens.len()]
                        == expected_target_tokens,
                    "real DFlash2 GPU target tile did not contain anchor plus proposals"
                );
                Ok(())
            })
            .expect("run real Qwen3.6 DFlash2 cycle");
    }

    fn capture_lane(
        values: &[f32],
        lane: usize,
        rows: usize,
        hidden: usize,
        layer_ids: &[usize],
    ) -> DFlashCaptureSession {
        let layers = layer_ids.len();
        let mut capture = DFlashCaptureSession::new(layer_ids.to_vec(), rows, hidden, false);
        for row in 0..rows {
            for layer in 0..layers {
                let source = ((lane * rows + row) * layers + layer) * hidden;
                let destination = (layer * rows + row) * hidden;
                capture.hidden_output[destination..destination + hidden]
                    .copy_from_slice(&values[source..source + hidden]);
            }
        }
        capture
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.6 GGUF/checkpoint, HF2Q_DFLASH2_REFERENCE_FIXTURES and Apple Metal"]
    fn qwen36_fixture_suite_matches_pinned_reference() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let gguf_path =
            PathBuf::from(std::env::var_os("HF2Q_QWEN36_GGUF").expect("HF2Q_QWEN36_GGUF"));
        let config_path = PathBuf::from(
            std::env::var_os("HF2Q_QWEN36_DFLASH2_CONFIG").expect("HF2Q_QWEN36_DFLASH2_CONFIG"),
        );
        let weights_path = PathBuf::from(
            std::env::var_os("HF2Q_QWEN36_DFLASH2_WEIGHTS").expect("HF2Q_QWEN36_DFLASH2_WEIGHTS"),
        );
        let fixture_directory = PathBuf::from(
            std::env::var_os("HF2Q_DFLASH2_REFERENCE_FIXTURES")
                .expect("HF2Q_DFLASH2_REFERENCE_FIXTURES"),
        );

        let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open Qwen3.6 GGUF");
        let mut progress = LoadProgress::new(false, 1, 0);
        let model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load Qwen3.6");
        model
            .ensure_gpu_cache_primed()
            .expect("prime Qwen3.6 GPU cache");
        model
            .with_gpu_cache_mut(|device, registry| {
                let config_bytes = std::fs::read(&config_path)?;
                let config = QWEN36_DFLASH2.parse_config(&config_bytes)?;
                let adapter = QwenDFlash2Adapter::new(&model, &config)?;
                let target_hidden = model.cfg.hidden_size as usize;

                for (case, file_name) in [
                    ("short", "qwen36-short-f32.safetensors"),
                    ("wrap", "qwen36-wrap-f32.safetensors"),
                    ("single", "qwen36-single1-f32.safetensors"),
                    ("multi", "qwen36-multi-f32.safetensors"),
                    ("restore", "qwen36-restore-f32.safetensors"),
                ] {
                    let fixture_path = fixture_directory.join(file_name);
                    let fixture_bytes = std::fs::read(&fixture_path)
                        .with_context(|| format!("read {case} reference fixture"))?;
                    let (_, metadata) = SafeTensors::read_metadata(&fixture_bytes)
                        .with_context(|| format!("read {case} reference metadata"))?;
                    let metadata = metadata
                        .metadata()
                        .as_ref()
                        .context("reference metadata is missing")?;
                    ensure!(
                        metadata.get("schema").map(String::as_str)
                            == Some("hf2q.dflash2-reference-fixture.v2"),
                        "{case} fixture schema differs"
                    );
                    ensure!(
                        metadata.get("reference_commit").map(String::as_str)
                            == Some("95c8aeca5e4b4c4f9c0c967c05ab89fa3ed24f4c"),
                        "{case} fixture reference commit differs"
                    );
                    ensure!(
                        metadata.get("compute_dtype").map(String::as_str) == Some("float32"),
                        "{case} fixture must match hf2q's float32 activations"
                    );
                    ensure!(
                        metadata.get("case").map(String::as_str) == Some(case),
                        "{case} fixture metadata differs"
                    );
                    ensure!(
                        metadata.get("attention_precision").map(String::as_str)
                            == Some("bfloat16"),
                        "{case} fixture must match hf2q's packed attention precision"
                    );
                    let context_start = metadata
                        .get("context_start")
                        .context("context_start metadata is missing")?
                        .parse::<usize>()?;
                    let context_rows = metadata
                        .get("context_rows")
                        .context("context_rows metadata is missing")?
                        .parse::<usize>()?;
                    let fixture = SafeTensors::deserialize(&fixture_bytes)
                        .with_context(|| format!("parse {case} fixture"))?;

                    let target_view = fixture.tensor("input.target_hidden")?;
                    let target_shape = target_view.shape();
                    ensure!(
                        target_shape.len() == 3
                            && target_shape[1] == context_rows
                            && target_shape[2] == config.fc_input_dim(),
                        "{case} target input shape {target_shape:?} differs"
                    );
                    let lanes = target_shape[0];
                    let target = float_values(target_view);
                    let mut runtime = DFlash2Runtime::load_verified(
                        device,
                        &QWEN36_DFLASH2,
                        &config_path,
                        &weights_path,
                        &adapter,
                        lanes,
                    )?;

                    if case == "restore" {
                        let initial_view = fixture.tensor("input.initial_target_hidden")?;
                        let initial_shape = initial_view.shape();
                        ensure!(
                            initial_shape.len() == 3
                                && initial_shape[0] == lanes
                                && initial_shape[2] == config.fc_input_dim(),
                            "restore initial target shape {initial_shape:?} differs"
                        );
                        let initial_rows = initial_shape[1];
                        let initial = float_values(initial_view);
                        for lane in 0..lanes {
                            let capture = capture_lane(
                                &initial,
                                lane,
                                initial_rows,
                                target_hidden,
                                &config.target_layer_ids,
                            );
                            runtime.ingest_target_capture(device, registry, &capture, lane, 0)?;
                        }
                    }
                    for lane in 0..lanes {
                        let capture = capture_lane(
                            &target,
                            lane,
                            context_rows,
                            target_hidden,
                            &config.target_layer_ids,
                        );
                        runtime.ingest_target_capture(
                            device,
                            registry,
                            &capture,
                            lane,
                            context_start,
                        )?;
                    }
                    ensure!(
                        runtime
                            .lane_lengths()
                            .iter()
                            .all(|length| *length == context_start + context_rows),
                        "{case} context lengths {:?} differ",
                        runtime.lane_lengths()
                    );

                    let anchors = i64_values(fixture.tensor("input.anchor")?)
                        .into_iter()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    let positions = runtime
                        .lane_lengths()
                        .iter()
                        .map(|length| u32::try_from(*length))
                        .collect::<Result<Vec<_>, _>>()?;
                    prepare_decode_positions(&mut runtime.workspace, &config, &positions)?;
                    let mut encoder = device.command_encoder()?;
                    dispatch_dflash2_input(
                        &runtime.tensors,
                        &mut encoder,
                        registry,
                        device,
                        &config,
                        &adapter,
                        &anchors,
                        &mut runtime.workspace,
                    )?;
                    encoder.commit_and_wait()?;
                    let expected_noise = float_values(fixture.tensor("input.noise_embedding")?);
                    let noise_delta = max_abs(
                        runtime.workspace.hidden[0].as_slice::<f32>()?,
                        &expected_noise,
                    );
                    ensure!(
                        noise_delta <= 0.03,
                        "{case} input embedding max_abs={noise_delta} exceeds 0.03"
                    );
                    for layer in 0..config.num_hidden_layers {
                        let mut encoder = device.command_encoder()?;
                        dispatch_layer(
                            &mut encoder,
                            registry,
                            device,
                            &config,
                            &runtime.tensors.layers[layer],
                            &runtime.workspace,
                            layer,
                        )?;
                        encoder.commit_and_wait()?;
                        let expected =
                            float_values(fixture.tensor(&format!("output.layer.{layer}.output"))?);
                        let delta = max_abs(
                            runtime.workspace.hidden[(layer + 1) & 1].as_slice::<f32>()?,
                            &expected,
                        );
                        let expected_abs_max = expected
                            .iter()
                            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
                        ensure!(
                            delta / expected_abs_max <= 0.05,
                            "{case} layer {layer} relative max_abs={} exceeds 0.05",
                            delta / expected_abs_max
                        );
                    }

                    runtime.run_draft(
                        device,
                        registry,
                        &adapter,
                        &anchors,
                        &positions,
                        &vec![0.0; lanes],
                        0,
                        &vec![8; lanes],
                        &vec![0.5; lanes * config.proposal_tokens()],
                    )?;

                    let expected_hidden = float_values(fixture.tensor("output.final_hidden")?);
                    let hidden_delta = max_abs(
                        runtime.workspace.final_hidden.as_slice::<f32>()?,
                        &expected_hidden,
                    );
                    ensure!(
                        hidden_delta <= 2.5,
                        "{case} final hidden max_abs={hidden_delta} exceeds 2.5"
                    );

                    let vocabulary = config.vocab_size;
                    let all_logits = runtime.workspace.logits.as_slice::<f32>()?;
                    let mut actual_logits =
                        Vec::with_capacity(lanes * config.proposal_tokens() * vocabulary);
                    for lane in 0..lanes {
                        let start = (lane * config.block_size + 1) * vocabulary;
                        actual_logits.extend_from_slice(
                            &all_logits[start..start + config.proposal_tokens() * vocabulary],
                        );
                    }
                    let expected_logits = float_values(fixture.tensor("output.logits")?);
                    let logits_delta = max_abs(&actual_logits, &expected_logits);
                    ensure!(
                        logits_delta <= 2.0,
                        "{case} logits max_abs={logits_delta} exceeds 2.0"
                    );

                    let all_selector = runtime.workspace.selector_hidden_bf16.as_slice::<u16>()?;
                    let mut actual_selector =
                        Vec::with_capacity(lanes * config.proposal_tokens() * config.selector_rank);
                    for lane in 0..lanes {
                        let start = (lane * config.block_size + 1) * config.selector_rank;
                        actual_selector.extend(
                            all_selector
                                [start..start + config.proposal_tokens() * config.selector_rank]
                                .iter()
                                .map(|bits| bf16::from_bits(*bits).to_f32()),
                        );
                    }
                    let expected_selector = float_values(fixture.tensor("output.selector_hidden")?);
                    let selector_delta = max_abs(&actual_selector, &expected_selector);
                    ensure!(
                        selector_delta <= 1.1,
                        "{case} selector hidden max_abs={selector_delta} exceeds 1.1"
                    );

                    let expected_candidates = i64_values(fixture.tensor("output.candidates")?)
                        .into_iter()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    let candidate_overlaps = runtime
                        .workspace
                        .candidates
                        .as_slice::<u32>()?
                        .chunks_exact(config.selector_top_k)
                        .zip(expected_candidates.chunks_exact(config.selector_top_k))
                        .map(|(actual, expected)| {
                            actual.iter().filter(|token| expected.contains(token)).count()
                        })
                        .collect::<Vec<_>>();
                    ensure!(
                        candidate_overlaps.iter().all(|overlap| *overlap >= 10),
                        "{case} candidate overlaps {candidate_overlaps:?} fall below 10/16"
                    );
                    let expected_path = i64_values(fixture.tensor("output.path")?)
                        .into_iter()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    let path_agreements = runtime
                        .proposals()?
                        .chunks_exact(config.proposal_tokens())
                        .zip(expected_path.chunks_exact(config.proposal_tokens()))
                        .map(|(actual, expected)| {
                            actual.iter().zip(expected).filter(|(a, e)| a == e).count()
                        })
                        .collect::<Vec<_>>();
                    ensure!(
                        path_agreements.iter().all(|agreement| *agreement >= 4),
                        "{case} path agreements {path_agreements:?} fall below 4/7: actual={:?} expected={expected_path:?}",
                        runtime.proposals()?
                    );
                }
                Ok(())
            })
            .expect("compare Qwen3.6 DFlash2 fixture suite");
    }
}
