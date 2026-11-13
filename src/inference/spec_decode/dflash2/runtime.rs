//! Verified checkpoint ownership and bounded DFlash2 cycle execution.

use std::path::Path;

use sha2::{Digest, Sha256};

#[cfg(test)]
use super::forward::dispatch_layer;
use super::{
    adapters::{dispatch_dflash2_input, target_vocabulary_head, DFlash2TargetFamily},
    config::DFlash2Config,
    forward::{
        dispatch_dflash2_context_commit, dispatch_dflash2_context_commit_at,
        dispatch_dflash2_forward, dispatch_dflash2_lane_ingest, prepare_decode_positions,
    },
    route_profile,
    schema::{DFlash2CheckpointDescriptor, DFlash2FamilyAdapter},
    selector::dispatch_dflash2_selection,
    staged::{reads_prepared_head, resolve_staged_draft, DFLASH2_DRAFT_STAGED_ENV},
    tensors::DFlash2ModelTensors,
    verify_budget::{DFlash2VerifyBudget, VerifyBudgetLane},
    weights::DFlash2WeightsFile,
    workspace::DFlash2Workspace,
};
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
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

/// A draft-context commit the GPU is executing
/// ([`DFlash2Runtime::submit_target_context`]).
pub(crate) struct SubmittedContextCommit {
    encoder: CommandEncoder,
}

/// Every input of one draft command. Two drafts with equal inputs, issued
/// over the same rings and verify-budget state, are the same command. Built
/// only when a prefetched command is stored.
#[derive(Clone, Debug)]
struct DraftInputs {
    anchors: Vec<u32>,
    logical_positions: Vec<u32>,
    temperature_bits: Vec<u32>,
    sampling_mask: u32,
    maximum_verify_rows: Vec<u32>,
    target_lane_order: Vec<u32>,
    uniform_bits: Vec<u32>,
    /// The caller reads the proposals on the host (the draft is waited for)
    /// rather than leaving the target tokens on the queue.
    wait: bool,
    /// [`DFlash2Runtime::budget_epoch`] when the command was encoded: the
    /// GPU row selection of an adaptive target reads the verify budget.
    budget_epoch: u64,
}

impl DraftInputs {
    /// Whether a draft call with these borrowed inputs is the command this
    /// snapshot describes. Floats compare by bits. Allocates nothing.
    #[allow(clippy::too_many_arguments)]
    fn matches(
        &self,
        anchors: &[u32],
        logical_positions: &[u32],
        temperatures: &[f32],
        sampling_mask: u32,
        maximum_verify_rows: &[u32],
        target_lane_order: &[u32],
        uniforms: &[f32],
        wait: bool,
        budget_epoch: u64,
    ) -> bool {
        let same_bits = |bits: &[u32], values: &[f32]| {
            bits.len() == values.len()
                && bits
                    .iter()
                    .zip(values)
                    .all(|(bits, value)| *bits == value.to_bits())
        };
        self.wait == wait
            && self.sampling_mask == sampling_mask
            && self.budget_epoch == budget_epoch
            && self.anchors == anchors
            && self.logical_positions == logical_positions
            && self.maximum_verify_rows == maximum_verify_rows
            && self.target_lane_order == target_lane_order
            && same_bits(&self.temperature_bits, temperatures)
            && same_bits(&self.uniform_bits, uniforms)
    }
}

/// A draft command committed ahead of the step that will use it
/// ([`DFlash2Runtime::prefetch_draft`]).
struct PrefetchedDraft {
    inputs: DraftInputs,
    encoder: CommandEncoder,
}

/// What became of the draft commands a runtime prefetched.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DraftPrefetchCounts {
    /// Committed ahead of their step.
    pub prefetched: u64,
    /// Taken by the next draft call because its inputs were the same.
    pub adopted: u64,
    /// Waited for and dropped: the next draft call had other inputs, or the
    /// rings were about to change under it.
    pub discarded: u64,
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
    /// The next step's draft, when the previous step committed it
    /// ([`Self::prefetch_draft`]). While it is `Some` the command may be
    /// executing: it reads the rings and writes the draft outputs of the
    /// workspace, so nothing may touch either before [`Self::settle_prefetch`].
    prefetched: Option<PrefetchedDraft>,
    prefetch_counts: DraftPrefetchCounts,
    /// Counts verify-budget observations ([`DraftInputs::budget_epoch`]).
    budget_epoch: u64,
    /// Copies of the workspace's row positions and ring slots that a
    /// context commit submitted ahead of the next draft reads
    /// ([`Self::submit_target_context_detached`]); allocated on first use.
    context_positions: Option<ContextPositions>,
}

/// `rope_positions` and `write_positions` of one context commit.
struct ContextPositions {
    rope: MlxBuffer,
    write: MlxBuffer,
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
        crate::inference::route_host::require_explicit(device)?;
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
        const STAGED: route_profile::Switch = route_profile::switch(DFLASH2_DRAFT_STAGED_ENV);
        const GPU_TARGET_TOKENS: route_profile::Switch =
            route_profile::switch(GPU_TARGET_TOKENS_ENV);
        let staged = route_profile::with_value(STAGED, |raw| {
            resolve_staged_draft(raw, &config, &tensors, lanes)
        })?;
        let workspace = DFlash2Workspace::new(device, &config, lanes, staged)
            .context("allocate DFlash2 workspace")?;
        let gpu_target_tokens = route_profile::with_value(GPU_TARGET_TOKENS, |raw| {
            resolve_gpu_target_tokens(descriptor, raw)
        })?;
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
            prefetched: None,
            prefetch_counts: DraftPrefetchCounts::default(),
            budget_epoch: 0,
            context_positions: None,
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
            false,
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
        prefetch: bool,
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
        // A draft that does not involve a prefetch (the stock route, and a
        // step with nothing outstanding) allocates nothing here: the
        // outstanding snapshot is compared with the borrowed inputs, and a
        // snapshot is built only when a prefetched command is stored.
        if let Some(prefetched) = self.prefetched.take() {
            if !prefetch
                && prefetched.inputs.matches(
                    anchors,
                    logical_positions,
                    temperatures,
                    sampling_mask,
                    maximum_verify_rows,
                    target_lane_order,
                    uniforms,
                    wait,
                    self.budget_epoch,
                )
            {
                // The command this call would encode is already on the
                // queue, issued over the rings and budget this call sees.
                self.prefetch_counts.adopted += 1;
                mlx_native::command_timeline::mark("draft.prefetch_adopted");
                if wait {
                    prefetched
                        .encoder
                        .wait_until_completed()
                        .context("execute prefetched DFlash2 draft cycle")?;
                }
                return Ok(());
            }
            self.discard_prefetched(prefetched)?;
        }
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
        let vocabulary_head = target_vocabulary_head(
            target,
            registry,
            device,
            reads_prepared_head(&self.workspace, &self.config),
        )?;
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
        if wait && !prefetch {
            encoder
                .commit_and_wait_labeled("adr067.dflash2_draft")
                .context("execute DFlash2 draft cycle")?;
        } else {
            encoder.commit_labeled("adr067.dflash2_draft");
        }
        if let Some(staged) = self.workspace.staged.as_mut() {
            staged.command_committed();
        }
        if prefetch {
            self.prefetch_counts.prefetched += 1;
            let inputs = DraftInputs {
                anchors: anchors.to_vec(),
                logical_positions: logical_positions.to_vec(),
                temperature_bits: temperatures.iter().map(|value| value.to_bits()).collect(),
                sampling_mask,
                maximum_verify_rows: maximum_verify_rows.to_vec(),
                target_lane_order: target_lane_order.to_vec(),
                uniform_bits: uniforms.iter().map(|value| value.to_bits()).collect(),
                wait,
                budget_epoch: self.budget_epoch,
            };
            self.prefetched = Some(PrefetchedDraft { inputs, encoder });
        }
        Ok(())
    }

    /// Commit the draft command of the step that follows the one being
    /// settled, without waiting for it. `host_tokens` says how that step
    /// will call for its draft: [`Self::run_draft`] (`true`) or
    /// [`Self::run_draft_for_gpu_target`] with `target_lane_order`.
    ///
    /// The next draft call adopts the command when its inputs are exactly
    /// these and neither the rings nor the verify budget changed in between
    /// (any ring write goes through [`Self::settle_prefetch`], and a budget
    /// observation changes the inputs' epoch); otherwise it waits for the
    /// command, drops its outputs and drafts again. A draft command writes
    /// only this workspace's draft outputs and scratch, never the rings, so
    /// a discarded one leaves no trace.
    ///
    /// The caller has finished with the current step's draft outputs
    /// (proposals, verify rows, confidences): the command overwrites them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefetch_draft<T: DFlash2TargetFamily>(
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
        host_tokens: bool,
    ) -> Result<()> {
        let every_lane;
        let target_lane_order = if host_tokens {
            every_lane = (0..self.workspace.max_lanes as u32).collect::<Vec<_>>();
            &every_lane
        } else {
            target_lane_order
        };
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
            host_tokens,
            true,
        )
    }

    /// Wait for a prefetched draft command and drop it. Called before
    /// anything writes the rings or encodes another command over this
    /// workspace; a no-op when nothing is prefetched.
    pub(crate) fn settle_prefetch(&mut self) -> Result<()> {
        match self.prefetched.take() {
            Some(prefetched) => self.discard_prefetched(prefetched),
            None => Ok(()),
        }
    }

    fn discard_prefetched(&mut self, prefetched: PrefetchedDraft) -> Result<()> {
        self.prefetch_counts.discarded += 1;
        mlx_native::command_timeline::mark("draft.prefetch_discarded");
        prefetched
            .encoder
            .wait_until_completed()
            .context("wait for a discarded DFlash2 draft prefetch")
    }

    /// Whether a prefetched draft command is on the queue, possibly executing.
    pub(crate) fn prefetch_outstanding(&self) -> bool {
        self.prefetched.is_some()
    }

    pub(crate) fn prefetch_counts(&self) -> DraftPrefetchCounts {
        self.prefetch_counts
    }

    /// The current step's proposals. A prefetched draft overwrites them, so
    /// they are unreadable from the prefetch until the next draft call.
    pub fn proposals(&self) -> Result<&[u32]> {
        self.ensure_draft_outputs_current()?;
        Ok(self.workspace.proposed_tokens.as_slice::<u32>()?)
    }

    pub fn verify_rows(&self) -> Result<&[u32]> {
        self.ensure_draft_outputs_current()?;
        Ok(self.workspace.verify_rows.as_slice::<u32>()?)
    }

    fn ensure_draft_outputs_current(&self) -> Result<()> {
        ensure!(
            self.prefetched.is_none(),
            "DFlash2 draft outputs read after the next draft was prefetched"
        );
        Ok(())
    }

    pub(crate) fn select_verify_rows(
        &mut self,
        lane_indices: &[usize],
        maximum_rows: &[u32],
    ) -> Result<()> {
        self.ensure_draft_outputs_current()?;
        let rows = self
            .verify_budget
            .select_rows(&self.workspace, lane_indices, maximum_rows)?;
        let selected = self.workspace.verify_rows.as_mut_slice::<u32>()?;
        for (&lane, rows) in lane_indices.iter().zip(rows) {
            selected[lane] = u32::try_from(rows)?;
        }
        Ok(())
    }

    /// Reads the current step's draft confidences from the workspace: call
    /// it before [`Self::prefetch_draft`] overwrites them.
    pub(crate) fn observe_verify_budget(
        &mut self,
        lanes: &[VerifyBudgetLane],
        milliseconds: f64,
    ) -> Result<()> {
        ensure!(
            self.prefetched.is_none(),
            "DFlash2 verify-budget observation after the next draft was prefetched"
        );
        self.budget_epoch = self.budget_epoch.wrapping_add(1);
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
    /// The capture is CPU layer-major. The rows are converted to the
    /// draft's row-major layout as they are staged and appended in lane
    /// passes, one command buffer for the chunk
    /// ([`ingest_capture_lane_passes`]); the ring bytes are those of the
    /// block commits made after target verification.
    pub fn ingest_target_capture(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        capture: &DFlashCaptureSession,
        lane: usize,
        logical_start: usize,
    ) -> Result<()> {
        self.ingest_target_capture_rows(
            device,
            registry,
            capture,
            lane,
            logical_start,
            0..capture.seq_len,
        )
    }

    /// [`Self::ingest_target_capture`] for the capture rows `rows` only,
    /// appended at `logical_start`. A prefill forward that crossed a prompt
    /// boundary appends the rows before it and the rows after it as two
    /// calls, as the two forwards would have.
    pub fn ingest_target_capture_rows(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        capture: &DFlashCaptureSession,
        lane: usize,
        logical_start: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<()> {
        ensure!(
            lane < self.workspace.max_lanes,
            "DFlash2 capture lane {lane} is out of range"
        );
        self.settle_prefetch()?;
        ensure!(
            logical_start == self.lane_lengths[lane],
            "DFlash2 capture starts at {logical_start}, lane {lane} is at {}",
            self.lane_lengths[lane]
        );
        let appended = rows.len();
        ingest_capture_lane_passes(
            device,
            registry,
            &self.config,
            &self.tensors,
            &self.workspace,
            capture,
            lane,
            logical_start,
            rows,
        )?;
        self.lane_lengths[lane] = logical_start
            .checked_add(appended)
            .context("DFlash2 lane length overflow")?;
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
        let mut encoder = self.encode_target_context(device, registry, retained_counts)?;
        encoder
            .commit_and_wait()
            .context("commit DFlash2 target context")?;
        self.advance_lane_lengths(retained_counts)
    }

    /// Encode the context commit of [`Self::commit_target_context`] and hand
    /// it to the GPU without waiting. The rings are being written until
    /// [`Self::finish_target_context`] returns: until then nothing may read
    /// or write them on the host, and the lane lengths are unchanged.
    ///
    /// Committing another command afterwards does not order it behind this
    /// one: with a concurrent compute encoder per command, a later command
    /// of the same queue was measured starting before this one ended in
    /// most cycles. A later command may therefore share no resource with
    /// the context commit (which reads the captured target rows, the
    /// retained counts and the ring positions and writes the draft rings),
    /// and only [`Self::finish_target_context`] says the rings are settled.
    pub(crate) fn submit_target_context(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        retained_counts: &[u32],
    ) -> Result<SubmittedContextCommit> {
        let mut encoder = self.encode_target_context(device, registry, retained_counts)?;
        encoder.commit();
        Ok(SubmittedContextCommit { encoder })
    }

    /// [`Self::submit_target_context`] for a step that commits the next
    /// draft behind it: the commit reads copies of the workspace's row
    /// positions and ring slots, so the next draft's positions can be
    /// written while it executes, and every lane's length moves past the
    /// rows it retains at once (the next draft is issued from those
    /// lengths). The rings are settled only when
    /// [`Self::finish_target_context`] (with `advance` false) returns.
    ///
    /// What orders the draft committed behind this command after it is not
    /// the queue as such (commands of one queue that share no resource
    /// overlap; the replay beside this commit does): it is Metal's
    /// dependency ordering of tracked resources. mlx-native allocates
    /// buffers with `newBufferWithLength` and shared storage, whose hazard
    /// tracking is the default tracked mode, and the draft rings (and the
    /// workspace scratch) are bound directly to both commands, so on an
    /// `MTLCommandQueue` the draft's reads of the rings are synchronized
    /// behind this command's writes. A ring moved to an untracked heap or
    /// reached through anything but a direct binding would lose that
    /// ordering.
    ///
    /// If the step fails after this call, clear its lanes
    /// ([`Self::settle_submitted_step`]): their lengths have moved.
    pub(crate) fn submit_target_context_detached(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        retained_counts: &[u32],
    ) -> Result<SubmittedContextCommit> {
        self.settle_prefetch()?;
        self.stage_retained_counts(retained_counts)?;
        if self.context_positions.is_none() {
            let copy = |source: &MlxBuffer| {
                device.alloc_buffer(
                    source.element_count() * std::mem::size_of::<u32>(),
                    mlx_native::DType::U32,
                    vec![source.element_count()],
                )
            };
            self.context_positions = Some(ContextPositions {
                rope: copy(&self.workspace.rope_positions)?,
                write: copy(&self.workspace.write_positions)?,
            });
        }
        let positions = self.context_positions.as_mut().expect("allocated above");
        positions
            .rope
            .as_mut_slice::<u32>()?
            .copy_from_slice(self.workspace.rope_positions.as_slice::<u32>()?);
        positions
            .write
            .as_mut_slice::<u32>()?
            .copy_from_slice(self.workspace.write_positions.as_slice::<u32>()?);
        let mut encoder = device
            .command_encoder()
            .context("create DFlash2 context encoder")?;
        dispatch_dflash2_context_commit_at(
            &mut encoder,
            registry,
            device,
            &self.config,
            &self.tensors,
            &self.workspace.retained_counts,
            &self.workspace,
            &positions.rope,
            &positions.write,
        )?;
        encoder.commit();
        self.advance_lane_lengths(retained_counts)?;
        Ok(SubmittedContextCommit { encoder })
    }

    /// Wait for a submitted context commit itself (a wait on a later
    /// command of the queue does not stand in for it). `advance` moves each
    /// lane's length past the rows it retained; pass `false` when the step
    /// the rows belong to failed after the submission, and then clear those
    /// lanes: a ring that has wrapped lost its oldest rows to the commit.
    pub(crate) fn finish_target_context(
        &mut self,
        submitted: SubmittedContextCommit,
        retained_counts: &[u32],
        advance: bool,
    ) -> Result<()> {
        submitted
            .encoder
            .wait_until_completed()
            .context("commit DFlash2 target context")?;
        if advance {
            self.advance_lane_lengths(retained_counts)?;
        }
        Ok(())
    }

    /// The context-first order of a step that settles acceptance on the
    /// host, from the submission on: run `replay` (the target's verify
    /// commit, which encodes, commits and waits for its own replay command)
    /// while the submitted context commit executes, then wait for the
    /// context command.
    ///
    /// The two commands overlap on the GPU in most cycles and either may
    /// finish first; nothing orders them. That is correct because they share
    /// no resource (the replay reads the verify step's replay storage and
    /// writes the target's DeltaNet state; the context commit reads the
    /// captured target rows and writes the draft rings) and because this
    /// function waits for the context command before it returns, whatever
    /// `replay` returned. That wait must not be removed: the replay's own
    /// wait does not cover the context command.
    ///
    /// Returns `replay`'s result and the context's. A failed submission is
    /// reported in the context's after `replay` ran, as in the stock order.
    /// When `replay` fails, the lane lengths do not advance and the lanes of
    /// `step_lanes` are cleared, because their rings hold rows of a step
    /// that did not commit: every lane is attempted, and a clear that fails
    /// is reported in the context's result (the first failure; later ones
    /// are logged) so the caller can stop instead of serving from a ring it
    /// could not reset.
    pub(crate) fn settle_context_first(
        &mut self,
        submitted: Result<SubmittedContextCommit>,
        retained_counts: &[u32],
        step_lanes: &[usize],
        replay: impl FnOnce() -> Result<()>,
    ) -> (Result<()>, Result<()>) {
        let committed = replay();
        let mut context = submitted.and_then(|submitted| {
            self.finish_target_context(submitted, retained_counts, committed.is_ok())
        });
        if committed.is_err() {
            for &lane in step_lanes {
                let cleared = self
                    .clear_lane(lane)
                    .with_context(|| format!("clear DFlash2 lane {lane} after a failed step"));
                match (cleared, context.is_ok()) {
                    (Ok(()), _) => {}
                    (Err(error), true) => context = Err(error),
                    (Err(error), false) => {
                        tracing::error!(error = %format!("{error:#}"), "DFlash2 lane cleanup failed");
                    }
                }
            }
        }
        (committed, context)
    }

    /// Close a step whose context commit
    /// ([`Self::submit_target_context_detached`]) and replay have both been
    /// waited for: `context` is the context command's result, `committed`
    /// whether the target's commit succeeded. A step with either failed
    /// must not leave anything of itself behind: a draft prefetched for the
    /// next step is waited for and dropped, and the lanes of `step_lanes`
    /// are cleared as in [`Self::settle_context_first`] (their lengths
    /// moved at the submission). Returns the context's result, or the
    /// first cleanup failure.
    pub(crate) fn settle_submitted_step(
        &mut self,
        committed: bool,
        mut context: Result<()>,
        step_lanes: &[usize],
    ) -> Result<()> {
        if committed && context.is_ok() {
            return context;
        }
        for &lane in step_lanes {
            let cleared = self
                .clear_lane(lane)
                .with_context(|| format!("clear DFlash2 lane {lane} after a failed step"));
            match (cleared, context.is_ok()) {
                (Ok(()), _) => {}
                (Err(error), true) => context = Err(error),
                (Err(error), false) => {
                    tracing::error!(error = %format!("{error:#}"), "DFlash2 lane cleanup failed");
                }
            }
        }
        context
    }

    fn encode_target_context(
        &mut self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        retained_counts: &[u32],
    ) -> Result<CommandEncoder> {
        // A context commit shares the workspace's scratch and position
        // buffers with a draft command and writes the rings it reads.
        self.settle_prefetch()?;
        self.stage_retained_counts(retained_counts)?;
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
        Ok(encoder)
    }

    fn stage_retained_counts(&mut self, retained_counts: &[u32]) -> Result<()> {
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
        Ok(())
    }

    fn advance_lane_lengths(&mut self, retained_counts: &[u32]) -> Result<()> {
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
        self.settle_prefetch()?;
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

/// Append capture rows `rows` to `lane`'s rings at `logical_start`.
///
/// A decode commit carries `block_size` rows of every lane, so appending a
/// prefill capture through it projects `block_size` rows per pass over the
/// draft's context weights. Here each pass carries the workspace's whole row
/// capacity (`max_lanes * block_size` rows) of the one lane
/// ([`dispatch_dflash2_lane_ingest`]), and the passes of a chunk share one
/// command buffer: their inputs are staged in buffers of their own, so no
/// pass waits for the host. The projections, norms and RoPE run over the
/// same row count as a decode commit, and the ring bytes are those of
/// `block_size`-row commits (`lane_passes_write_the_rings_of_block_commits`).
#[allow(clippy::too_many_arguments)]
pub(super) fn ingest_capture_lane_passes(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    config: &DFlash2Config,
    tensors: &DFlash2ModelTensors,
    workspace: &DFlash2Workspace,
    capture: &DFlashCaptureSession,
    lane: usize,
    logical_start: usize,
    rows: std::ops::Range<usize>,
) -> Result<()> {
    ensure!(
        capture.target_layer_ids == config.target_layer_ids,
        "DFlash2 capture layers {:?} != checkpoint layers {:?}",
        capture.target_layer_ids,
        config.target_layer_ids
    );
    let layers = capture.target_layer_ids.len();
    let hidden = capture.hidden_size;
    let feature_width = config.fc_input_dim();
    ensure!(
        rows.start <= rows.end
            && rows.end <= capture.seq_len
            && layers * hidden == feature_width
            && capture.hidden_output.len() == layers * capture.seq_len * hidden,
        "DFlash2 capture rows {rows:?} of a {}-row, {layers}-layer, {hidden}-wide capture \
         do not fit the checkpoint's {feature_width}-wide context",
        capture.seq_len
    );
    let appended = rows.len();
    if appended == 0 {
        return Ok(());
    }
    let block = config.block_size;
    let groups = workspace.max_lanes;
    let pass_rows = groups * block;
    let passes = appended.div_ceil(pass_rows);
    let staged_rows = passes * pass_rows;
    ensure!(
        u32::try_from(logical_start + staged_rows).is_ok(),
        "DFlash2 context position exceeds u32"
    );

    // The rows in the draft's `[row][layer][hidden]` layout; the capture is
    // layer-major. Rows past the capture's end stay zero and are not committed.
    let mut captured = device.alloc_buffer(
        staged_rows * feature_width * std::mem::size_of::<f32>(),
        mlx_native::DType::F32,
        vec![staged_rows, feature_width],
    )?;
    {
        let staged = captured.as_mut_slice::<f32>()?;
        for (index, row) in rows.clone().enumerate() {
            for layer in 0..layers {
                let source = (layer * capture.seq_len + row) * hidden;
                let destination = (index * layers + layer) * hidden;
                staged[destination..destination + hidden]
                    .copy_from_slice(&capture.hidden_output[source..source + hidden]);
            }
        }
    }
    let words = |values: &[u32]| -> Result<MlxBuffer> {
        let mut buffer = device.alloc_buffer(
            std::mem::size_of_val(values),
            mlx_native::DType::U32,
            vec![values.len()],
        )?;
        buffer.as_mut_slice::<u32>()?.copy_from_slice(values);
        Ok(buffer)
    };
    let positions = words(
        &(0..staged_rows)
            .map(|index| (logical_start + index) as u32)
            .collect::<Vec<_>>(),
    )?;
    // Group `g` of pass `p` holds rows `first..first + block`.
    let group_firsts = (0..passes * groups).map(|group| group * block);
    let write_positions = words(
        &group_firsts
            .clone()
            .map(|first| ((logical_start + first) % config.sliding_window) as u32)
            .collect::<Vec<_>>(),
    )?;
    let commit_counts = words(
        &group_firsts
            .map(|first| appended.saturating_sub(first).min(block) as u32)
            .collect::<Vec<_>>(),
    )?;

    let mut encoder = device
        .command_encoder()
        .context("create DFlash2 capture ingest encoder")?;
    let word = std::mem::size_of::<u32>();
    for pass in 0..passes {
        dispatch_dflash2_lane_ingest(
            &mut encoder,
            registry,
            device,
            config,
            tensors,
            workspace,
            lane,
            &captured
                .slice_view(
                    (pass * pass_rows * feature_width * std::mem::size_of::<f32>()) as u64,
                    pass_rows * feature_width,
                )
                .with_shape(vec![pass_rows, feature_width])?,
            &positions.slice_view((pass * pass_rows * word) as u64, pass_rows),
            &write_positions.slice_view((pass * groups * word) as u64, groups),
            &commit_counts.slice_view((pass * groups * word) as u64, groups),
        )
        .with_context(|| format!("encode DFlash2 capture pass {pass}"))?;
    }
    encoder
        .commit_and_wait()
        .context("commit DFlash2 capture ingest")?;
    Ok(())
}

#[cfg(test)]
#[path = "context_commit_tests.rs"]
mod context_commit_tests;

#[cfg(test)]
#[path = "draft_prefetch_tests.rs"]
mod draft_prefetch_tests;

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

    /// The capture ingest the lane passes replaced: one decode-shaped commit
    /// of `block_size` rows on the lane's own row group, one command buffer
    /// each. Kept as the reference of the gate below.
    fn ingest_block_commits(
        runtime: &mut DFlash2Runtime,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        capture: &DFlashCaptureSession,
        lane: usize,
    ) -> Result<()> {
        use crate::inference::spec_decode::dflash::hidden_capture::extract_drafter_concat;
        let concatenated = extract_drafter_concat(
            &capture.hidden_output,
            &capture.target_layer_ids,
            &runtime.config.target_layer_ids,
            capture.seq_len,
            capture.hidden_size,
        )?;
        let width = runtime.config.fc_input_dim();
        let block = runtime.config.block_size;
        for chunk in concatenated.chunks(block * width) {
            let lane_start = lane * block * width;
            let destination = runtime
                .workspace
                .captured_target_hidden
                .as_mut_slice::<f32>()?;
            destination[lane_start..lane_start + block * width].fill(0.0);
            destination[lane_start..lane_start + chunk.len()].copy_from_slice(chunk);
            let positions: Vec<u32> = runtime
                .lane_lengths
                .iter()
                .map(|length| *length as u32)
                .collect();
            prepare_decode_positions(&mut runtime.workspace, &runtime.config, &positions)?;
            let mut retained = vec![0_u32; runtime.workspace.max_lanes];
            retained[lane] = (chunk.len() / width) as u32;
            runtime.commit_target_context(device, registry, &retained)?;
        }
        Ok(())
    }

    /// Real-checkpoint gate for the lane-pass capture ingest, run explicitly
    /// with the served draft switches on one model:
    /// `HF2Q_QWEN38_GGUF=… HF2Q_DFLASH2_CONFIG=… HF2Q_DFLASH2_WEIGHTS=… HF2Q_DFLASH2_DRAFT_AF4_G64=1
    /// HF2Q_DFLASH2_DRAFT_STAGED=1 <test binary> capture_ingest_matches_block_commits_on_the_real_checkpoint
    /// --ignored --nocapture --test-threads=1` (or `HF2Q_QWEN36_GGUF=…`).
    ///
    /// At one to four lanes, every lane receives prefill chunks of 5, 31,
    /// 157 and 720 rows (so chunks start off a block boundary and end in a
    /// partial block) and then enough rows to wrap its ring, once through
    /// the block commits and once through the lane passes. After every chunk
    /// the rings of all lanes and the lane cursors must agree bit for bit,
    /// and so must the logits, proposals and verify rows of a draft cycle
    /// run on the context each route left.
    #[test]
    #[ignore = "requires HF2Q_QWEN38_GGUF or HF2Q_QWEN36_GGUF, HF2Q_DFLASH2_CONFIG, HF2Q_DFLASH2_WEIGHTS and Apple Metal; run alone"]
    fn capture_ingest_matches_block_commits_on_the_real_checkpoint() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let (gguf_path, descriptor) = match (
            std::env::var_os("HF2Q_QWEN38_GGUF"),
            std::env::var_os("HF2Q_QWEN36_GGUF"),
        ) {
            (Some(path), None) => (PathBuf::from(path), &QWEN38_DFLASH2),
            (None, Some(path)) => (PathBuf::from(path), &QWEN36_DFLASH2),
            _ => panic!("set exactly one of HF2Q_QWEN38_GGUF and HF2Q_QWEN36_GGUF"),
        };
        let config_path =
            PathBuf::from(std::env::var_os("HF2Q_DFLASH2_CONFIG").expect("HF2Q_DFLASH2_CONFIG"));
        let weights_path =
            PathBuf::from(std::env::var_os("HF2Q_DFLASH2_WEIGHTS").expect("HF2Q_DFLASH2_WEIGHTS"));
        let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open target GGUF");
        let mut progress = LoadProgress::new(false, 1, 0);
        let model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load target");
        model.ensure_gpu_cache_primed().expect("prime GPU cache");

        const CHUNKS: [usize; 4] = [5, 31, 157, 720];
        fn random(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
        }
        fn rings(runtime: &DFlash2Runtime) -> Vec<u16> {
            runtime
                .workspace
                .ring_keys
                .iter()
                .chain(&runtime.workspace.ring_values)
                .flat_map(|ring| ring.as_slice::<u16>().expect("ring bits").to_vec())
                .collect()
        }

        model
            .with_gpu_cache_mut(|device, registry| {
                let config_bytes = std::fs::read(&config_path)?;
                let config = descriptor.parse_config(&config_bytes)?;
                let adapter = QwenDFlash2Adapter::new(&model, &config)?;
                let layers = config.target_layer_ids.len();
                let hidden = config.fc_input_dim() / layers;
                // The last chunk carries every lane past one ring wrap.
                let prefix: usize = CHUNKS.iter().sum();
                let mut chunks = CHUNKS.to_vec();
                chunks.push(config.sliding_window + 13 - prefix.min(config.sliding_window));
                for lanes in 1..=4_usize {
                    let mut runtime = DFlash2Runtime::load_verified(
                        device,
                        descriptor,
                        &config_path,
                        &weights_path,
                        &adapter,
                        lanes,
                    )?;
                    let proposals = runtime.config.proposal_tokens();
                    // One route's whole run: the rings and cursors after
                    // every chunk, then a draft cycle's outputs.
                    let run = |runtime: &mut DFlash2Runtime,
                                   registry: &mut KernelRegistry,
                                   lane_passes: bool|
                     -> Result<(Vec<Vec<u16>>, Vec<usize>, Vec<u32>, Vec<u32>, Vec<u32>)> {
                        for lane in 0..lanes {
                            runtime.clear_lane(lane)?;
                        }
                        let mut after_chunks = Vec::new();
                        let mut state = 0x5EED_u64;
                        for &rows in &chunks {
                            for lane in 0..lanes {
                                let mut capture = DFlashCaptureSession::new(
                                    runtime.config.target_layer_ids.clone(),
                                    rows,
                                    hidden,
                                    false,
                                );
                                for value in &mut capture.hidden_output {
                                    *value = 4.0 * random(&mut state);
                                }
                                let start = runtime.lane_lengths()[lane];
                                if lane_passes {
                                    runtime.ingest_target_capture(
                                        device, registry, &capture, lane, start,
                                    )?;
                                } else {
                                    ingest_block_commits(
                                        runtime, device, registry, &capture, lane,
                                    )?;
                                }
                                ensure!(
                                    runtime.lane_lengths()[lane] == start + rows,
                                    "lane {lane} cursor after a {rows}-row chunk"
                                );
                            }
                            after_chunks.push(rings(runtime));
                        }
                        let lengths = runtime.lane_lengths().to_vec();
                        let positions: Vec<u32> =
                            lengths.iter().map(|length| *length as u32).collect();
                        runtime.run_draft(
                            device,
                            registry,
                            &adapter,
                            &vec![1; lanes],
                            &positions,
                            &vec![0.0; lanes],
                            0,
                            &vec![8; lanes],
                            &vec![0.5; lanes * proposals],
                        )?;
                        Ok((
                            after_chunks,
                            lengths,
                            runtime.proposals()?.to_vec(),
                            runtime.verify_rows()?.to_vec(),
                            runtime
                                .workspace
                                .logits
                                .as_slice::<u32>()?
                                .to_vec(),
                        ))
                    };
                    let started = std::time::Instant::now();
                    let blocks = run(&mut runtime, registry, false)?;
                    let blocks_ms = started.elapsed().as_secs_f64() * 1e3;
                    let started = std::time::Instant::now();
                    let passes = run(&mut runtime, registry, true)?;
                    let passes_ms = started.elapsed().as_secs_f64() * 1e3;
                    eprintln!(
                        "[DFLASH2_INGEST] {} lanes={lanes}: {} rows per lane and one draft cycle: block commits {blocks_ms:.1} ms, lane passes {passes_ms:.1} ms",
                        descriptor.id,
                        chunks.iter().sum::<usize>()
                    );
                    for (index, (block, pass)) in blocks.0.iter().zip(&passes.0).enumerate() {
                        ensure!(block.iter().any(|&bits| bits != 0), "nothing was appended");
                        let differing = block.iter().zip(pass).filter(|(a, b)| a != b).count();
                        ensure!(
                            differing == 0,
                            "{lanes} lanes, after the {}-row chunk: {differing} of {} ring values differ",
                            chunks[index],
                            block.len()
                        );
                    }
                    ensure!(blocks.1 == passes.1, "{lanes} lanes: lane cursors differ");
                    ensure!(blocks.2 == passes.2, "{lanes} lanes: draft proposals differ");
                    ensure!(blocks.3 == passes.3, "{lanes} lanes: verify rows differ");
                    ensure!(blocks.4 == passes.4, "{lanes} lanes: draft logits differ");
                    eprintln!(
                        "[DFLASH2_INGEST] {} lanes={lanes} chunks={chunks:?} staged={} rings, cursors and draft outputs identical",
                        descriptor.id,
                        runtime.workspace.staged.is_some()
                    );
                }
                Ok(())
            })
            .expect("lane-pass ingest against block commits");
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
