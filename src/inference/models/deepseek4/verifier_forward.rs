//! Full one-token DeepSeek-V4 verifier execution and cache publication.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, GraphExecutor, IdMmScratch, MlxBuffer, MM_ID_ROUTING_THRESHOLD};
use std::sync::atomic::{AtomicBool, Ordering};

use super::cache::{CacheSpan, Deepseek4Cache};
use super::forward_support::{
    alloc, alloc_persistent, begin_decode_pool_token, begin_prefill_pool_layer,
    begin_prefill_submission_inputs, end_decode_pool_token, end_prefill_pool_layer,
    end_prefill_submission_inputs,
};
use super::submission::{drain, retained_reference_pipeline_enabled, SubmissionChain};
use super::Deepseek4Model;

/// Keep the native 128-token raw window as the alignment unit, but batch
/// enough windows to amortize the 43-layer command-buffer and mask setup.
const DEFAULT_MATRIX_PREFILL_WINDOWS: usize = 32;
/// The 100 GiB mixed agent profile needs a 2K transaction to remain below
/// Metal's working-set ceiling. Smaller artifacts retain the historical 4K.
const LARGE_MODEL_MATRIX_PREFILL_WINDOWS: usize = 16;
const LARGE_MODEL_RESIDENT_BYTES: u64 = 100_000_000_000;
pub(crate) const MIN_MATRIX_APPEND_TOKENS: usize = 33;
pub(crate) const MAX_COOPERATIVE_PREFILL_ROWS: usize = 2_048;

struct CooperativePrefillLayout {
    rows_per_sequence: Vec<usize>,
    total_rows: usize,
    row_elements: usize,
    combined_elements: usize,
}

static STAGE_PROFILE_CAPTURED: AtomicBool = AtomicBool::new(false);

pub(super) fn publish_state_after_gate<State>(
    state: &mut State,
    commit_gate: impl FnOnce() -> Result<()>,
    publish: impl FnOnce(&mut State) -> Result<()>,
    poison: impl FnOnce(&mut State),
    rejection_context: &'static str,
    publication_context: &'static str,
) -> Result<()> {
    if let Err(error) = commit_gate() {
        poison(state);
        return Err(error).context(rejection_context);
    }
    if let Err(error) = publish(state) {
        poison(state);
        return Err(error).context(publication_context);
    }
    Ok(())
}

fn publish_prefill_cache_after_gate(
    cache: &mut Deepseek4Cache,
    start_position: usize,
    token_count: usize,
    commit_gate: impl FnOnce() -> Result<()>,
) -> Result<()> {
    publish_state_after_gate(
        cache,
        commit_gate,
        |cache| {
            cache
                .commit_prefill(start_position, token_count)
                .map_err(Into::into)
        },
        Deepseek4Cache::poison,
        "DeepSeek-V4 prompt prefill rejected before cache publication",
        "publish complete DeepSeek-V4 prompt prefill",
    )
}

pub(super) fn publish_prefill_cohort_after_gate(
    caches: &mut [&mut Deepseek4Cache],
    spans: &[CacheSpan],
    commit_gate: impl FnOnce() -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(
        caches.len() == spans.len(),
        "DeepSeek-V4 cooperative prefill received {} cache spans for {} caches",
        spans.len(),
        caches.len()
    );
    let tickets = spans
        .iter()
        .zip(caches.iter())
        .map(|(span, cache)| {
            cache
                .validate_prefill_commit(span.start_position, span.token_count)
                .context("prevalidate DeepSeek-V4 cooperative prefill publication")
        })
        .collect::<Result<Vec<_>>>();
    let tickets = match tickets {
        Ok(tickets) => tickets,
        Err(error) => {
            for cache in caches.iter_mut() {
                cache.poison();
            }
            return Err(error);
        }
    };
    if let Err(error) = commit_gate() {
        for cache in caches.iter_mut() {
            cache.poison();
        }
        return Err(error)
            .context("DeepSeek-V4 cooperative prefill rejected before cache publication");
    }
    for (cache, ticket) in caches.iter_mut().zip(tickets) {
        cache.publish_prefill_end(ticket);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn plan_cooperative_prefill_layout(
    token_batches: &[&[u32]],
    hyper_connection_count: usize,
    hidden_size: usize,
) -> Result<()> {
    cooperative_prefill_layout(token_batches, hyper_connection_count, hidden_size).map(|_| ())
}

fn cooperative_prefill_layout(
    token_batches: &[&[u32]],
    hyper_connection_count: usize,
    hidden_size: usize,
) -> Result<CooperativePrefillLayout> {
    let rows_per_sequence = token_batches
        .iter()
        .map(|tokens| tokens.len())
        .collect::<Vec<_>>();
    let total_rows = rows_per_sequence.iter().try_fold(0_usize, |total, rows| {
        total
            .checked_add(*rows)
            .context("DeepSeek-V4 cooperative prefill row count overflow")
    })?;
    anyhow::ensure!(
        total_rows <= MAX_COOPERATIVE_PREFILL_ROWS,
        "DeepSeek-V4 cooperative prefill has {total_rows} aggregate rows; maximum is {MAX_COOPERATIVE_PREFILL_ROWS}"
    );
    let row_elements = hyper_connection_count
        .checked_mul(hidden_size)
        .context("DeepSeek-V4 cooperative prefill row width overflow")?;
    let combined_elements = total_rows
        .checked_mul(row_elements)
        .context("DeepSeek-V4 cooperative prefill state size overflow")?;
    Ok(CooperativePrefillLayout {
        rows_per_sequence,
        total_rows,
        row_elements,
        combined_elements,
    })
}

fn publish_verifier_cache_after_gate(
    cache: &mut Deepseek4Cache,
    position: usize,
    commit_gate: impl FnOnce() -> Result<()>,
) -> Result<()> {
    publish_state_after_gate(
        cache,
        commit_gate,
        |cache| cache.commit_step(position).map_err(Into::into),
        Deepseek4Cache::poison,
        "DeepSeek-V4 verifier token rejected before cache publication",
        "publish complete DeepSeek-V4 verifier token",
    )
}

pub(crate) fn matrix_prefill_chunk_len(
    cache_position: usize,
    remaining: usize,
    sliding_window: usize,
    window_multiplier: usize,
) -> usize {
    if remaining == 0 || sliding_window == 0 || window_multiplier == 0 {
        return 0;
    }
    if cache_position > 0 && remaining < MIN_MATRIX_APPEND_TOKENS {
        return 0;
    }
    remaining.min(sliding_window.saturating_mul(window_multiplier))
}

fn prefill_windows_for_resident_bytes(resident_bytes: u64) -> usize {
    if resident_bytes >= LARGE_MODEL_RESIDENT_BYTES {
        LARGE_MODEL_MATRIX_PREFILL_WINDOWS
    } else {
        DEFAULT_MATRIX_PREFILL_WINDOWS
    }
}

fn graph_reorder_enabled() -> Result<bool> {
    let Some(value) = std::env::var_os("HF2Q_DEEPSEEK_GRAPH_REORDER") else {
        return Ok(true);
    };
    match value
        .to_str()
        .context("HF2Q_DEEPSEEK_GRAPH_REORDER is not valid UTF-8")?
    {
        "0" => Ok(false),
        "1" => Ok(true),
        value => anyhow::bail!("HF2Q_DEEPSEEK_GRAPH_REORDER must be 0 or 1 (got {value})"),
    }
}

fn graph_layers_per_command_buffer(
    layers: usize,
    graph_reorder: bool,
    requires_single_layer: bool,
) -> Result<usize> {
    let Some(value) = std::env::var_os("HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB") else {
        return Ok(default_graph_layers_per_command_buffer(
            layers,
            graph_reorder,
            requires_single_layer,
        ));
    };
    let value = value
        .to_str()
        .context("HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB is not valid UTF-8")?
        .parse::<usize>()
        .context("HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB must be a positive integer")?;
    anyhow::ensure!(
        (1..=layers).contains(&value),
        "HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB must be in 1..={layers}"
    );
    Ok(value)
}

fn default_graph_layers_per_command_buffer(
    layers: usize,
    graph_reorder: bool,
    requires_single_layer: bool,
) -> usize {
    if graph_reorder && !requires_single_layer {
        layers.min(4)
    } else {
        1
    }
}

impl Deepseek4Model {
    pub(crate) fn matrix_prefill_window_multiplier(&self) -> Result<usize> {
        let value = match std::env::var("HF2Q_DEEPSEEK_PREFILL_WINDOWS") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => {
                return Ok(prefill_windows_for_resident_bytes(
                    self.weights.resident_bytes(),
                ));
            }
            Err(error) => {
                return Err(error).context("HF2Q_DEEPSEEK_PREFILL_WINDOWS is not valid UTF-8");
            }
        };
        let value = value
            .parse::<usize>()
            .context("HF2Q_DEEPSEEK_PREFILL_WINDOWS must be a positive integer")?;
        anyhow::ensure!(
            value > 0,
            "HF2Q_DEEPSEEK_PREFILL_WINDOWS must be a positive integer"
        );
        Ok(value)
    }

    /// Execute one bounded prompt chunk layer-major with true matrix rows.
    pub fn forward_verifier_prefill(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        self.forward_verifier_prefill_with_commit_gate(token_ids, cache, || Ok(()))
    }

    /// Execute independent prompt transactions as one layer-major cohort.
    ///
    /// Attention and cache writes remain sequence-local. After each sequence's
    /// attention finishes, its rows are packed into one contiguous tensor and
    /// the row-local FFN/MoE executes once across the complete cohort. Logical
    /// cache cursors publish only after every GPU transaction and the shared
    /// commit gate complete successfully.
    pub(crate) fn forward_verifier_prefill_cohort_with_commit_gate(
        &mut self,
        token_batches: &[&[u32]],
        caches: &mut [&mut Deepseek4Cache],
        commit_gate: impl FnOnce() -> Result<()>,
    ) -> Result<Vec<MlxBuffer>> {
        anyhow::ensure!(
            token_batches.len() >= 2,
            "DeepSeek-V4 cooperative prefill requires at least two sequences"
        );
        anyhow::ensure!(
            token_batches.len() == caches.len(),
            "DeepSeek-V4 cooperative prefill received {} token batches for {} caches",
            token_batches.len(),
            caches.len()
        );
        anyhow::ensure!(
            token_batches.iter().all(|tokens| !tokens.is_empty()),
            "DeepSeek-V4 cooperative prefill received an empty token batch"
        );
        let layout = cooperative_prefill_layout(
            token_batches,
            self.cfg.hyper_connection_count as usize,
            self.cfg.hidden_size as usize,
        )?;
        let spans = token_batches
            .iter()
            .zip(caches.iter())
            .map(|(tokens, cache)| {
                cache
                    .plan_prefill(tokens.len())
                    .context("plan DeepSeek-V4 cooperative prefill transaction")
            })
            .collect::<Result<Vec<_>>>()?;
        let (states, submitted_any) =
            self.forward_verifier_prefill_cohort_uncommitted(token_batches, caches, &spans, layout);
        let states = match states {
            Ok(states) => states,
            Err(error) => {
                if submitted_any {
                    for cache in caches.iter_mut() {
                        cache.poison();
                    }
                    return Err(error).context(
                        "DeepSeek-V4 cooperative prefill partially executed; caches poisoned",
                    );
                }
                return Err(error)
                    .context("DeepSeek-V4 cooperative prefill failed before GPU submission");
            }
        };
        publish_prefill_cohort_after_gate(caches, &spans, commit_gate)?;
        Ok(states)
    }

    /// Immediate-commit wrapper for diagnostics and parity tests.
    #[cfg(test)]
    pub(super) fn forward_verifier_prefill_cohort(
        &mut self,
        token_batches: &[&[u32]],
        caches: &mut [&mut Deepseek4Cache],
    ) -> Result<Vec<MlxBuffer>> {
        self.forward_verifier_prefill_cohort_with_commit_gate(token_batches, caches, || Ok(()))
    }

    /// Execute a bounded prompt transaction, but publish its logical cache
    /// cursor only after the caller accepts the completed GPU transaction.
    /// Serving uses this gate for the worker deadline lease; CLI and existing
    /// inference callers retain the historical immediate-commit wrapper.
    pub fn forward_verifier_prefill_with_commit_gate(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
        commit_gate: impl FnOnce() -> Result<()>,
    ) -> Result<MlxBuffer> {
        let span = cache
            .plan_prefill(token_ids.len())
            .context("plan DeepSeek-V4 batched prefill transaction")?;
        let profile_stages = std::env::var("HF2Q_DEEPSEEK_COMPRESSED_STAGE_PROFILE").as_deref()
            == Ok("1")
            && std::env::var("MLX_PROFILE_CB").as_deref() == Ok("1");
        if profile_stages {
            mlx_native::kernel_profile::reset();
        }
        let result = self.forward_verifier_prefill_uncommitted(token_ids, cache, &span);
        if profile_stages {
            eprintln!(
                "DeepSeek-V4 compressed prefill GPU stages at position {} for {} rows:",
                span.start_position, span.token_count,
            );
            for (label, entry) in mlx_native::kernel_profile::dump() {
                eprintln!(
                    "  {label}: {:.3} ms total over {} layers (min {:.3} ms; max {:.3} ms)",
                    entry.total_ns as f64 / 1e6,
                    entry.count,
                    entry.min_ns as f64 / 1e6,
                    entry.max_ns as f64 / 1e6,
                );
            }
        }
        match result {
            Ok(state) => {
                publish_prefill_cache_after_gate(
                    cache,
                    span.start_position,
                    span.token_count,
                    commit_gate,
                )?;
                Ok(state)
            }
            Err(error) => {
                cache.poison();
                Err(error).context("DeepSeek-V4 prefill partially executed; cache poisoned")
            }
        }
    }

    /// Prefill an arbitrarily long prompt in bounded matrix chunks. Small
    /// cached suffixes retain the exact incremental verifier; long suffixes
    /// use nonzero-position matrix transactions so prompt ingestion never
    /// degenerates into decode-style replay.
    pub fn forward_verifier_prompt(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        anyhow::ensure!(!token_ids.is_empty(), "DeepSeek-V4 prompt is empty");
        let profile_timing = std::env::var_os("HF2Q_DEEPSEEK_PREFILL_TIMING").is_some();
        let window_multiplier = self.matrix_prefill_window_multiplier()?;
        let mut state = None;
        let mut offset = 0;
        let prompt_start = std::time::Instant::now();
        while offset < token_ids.len() {
            let chunk = matrix_prefill_chunk_len(
                cache.position(),
                token_ids.len() - offset,
                self.cfg.sliding_window as usize,
                window_multiplier,
            );
            if chunk == 0 {
                break;
            }
            let chunk_start = std::time::Instant::now();
            let position = cache.position();
            state = Some(self.forward_verifier_prefill(&token_ids[offset..offset + chunk], cache)?);
            offset += chunk;
            if profile_timing {
                eprintln!(
                    "DeepSeek-V4 prefill chunk: position {position}; rows {chunk}; total {:.3} ms; cumulative {:.3} ms",
                    chunk_start.elapsed().as_secs_f64() * 1e3,
                    prompt_start.elapsed().as_secs_f64() * 1e3,
                );
            }
        }
        for &token in &token_ids[offset..] {
            state = Some(self.forward_verifier_one(token, cache)?);
        }
        state.context("DeepSeek-V4 prompt encoded zero chunks")
    }

    fn forward_verifier_prefill_cohort_uncommitted(
        &mut self,
        token_batches: &[&[u32]],
        caches: &mut [&mut Deepseek4Cache],
        spans: &[CacheSpan],
        layout: CooperativePrefillLayout,
    ) -> (Result<Vec<MlxBuffer>>, bool) {
        let mut submitted_any = false;
        let result = (|| -> Result<Vec<MlxBuffer>> {
            let layers = self.cfg.num_hidden_layers as usize;
            let CooperativePrefillLayout {
                rows_per_sequence,
                total_rows,
                row_elements,
                combined_elements,
            } = layout;
            let hc = self.cfg.hyper_connection_count as usize;
            let hidden = self.cfg.hidden_size as usize;
            let combined_tokens = token_batches
                .iter()
                .flat_map(|tokens| tokens.iter().copied())
                .collect::<Vec<_>>();
            let device = self.ctx.device().clone();
            let executor = GraphExecutor::new(device.clone());
            let reusable_states = [
                alloc_persistent(
                    &device,
                    DType::F32,
                    vec![total_rows, hc, hidden],
                    "cooperative prefill state ping",
                )?,
                alloc_persistent(
                    &device,
                    DType::F32,
                    vec![total_rows, hc, hidden],
                    "cooperative prefill state pong",
                )?,
            ];
            let combined_attention = alloc_persistent(
                &device,
                DType::F32,
                vec![total_rows, hc, hidden],
                "cooperative prefill attention",
            )?;
            let mut id_mm_scratch = if total_rows > MM_ID_ROUTING_THRESHOLD as usize {
                let rows = u32::try_from(total_rows)
                    .context("DeepSeek-V4 cooperative prefill rows exceed u32")?;
                Some([
                    IdMmScratch::alloc(self.ctx.device(), self.cfg.num_experts, rows)
                        .context("allocate cooperative gate/down mm_id scratch")?,
                    IdMmScratch::alloc(self.ctx.device(), self.cfg.num_experts, rows)
                        .context("allocate cooperative up mm_id scratch")?,
                ])
            } else {
                None
            };
            let mut state = None;
            for layer in 0..layers {
                let mut row_offset = 0_usize;
                for sequence in 0..token_batches.len() {
                    let rows = rows_per_sequence[sequence];
                    let element_offset = row_offset
                        .checked_mul(row_elements)
                        .context("DeepSeek-V4 cooperative state offset overflow")?;
                    let element_count = rows
                        .checked_mul(row_elements)
                        .context("DeepSeek-V4 cooperative state slice overflow")?;
                    begin_prefill_pool_layer();
                    let attention_result: Result<()> = (|| {
                        let mut session = executor.begin().with_context(|| {
                            format!(
                                "begin DeepSeek-V4 cooperative prefill layer {layer} sequence {sequence} attention"
                            )
                        })?;
                        // Several DeepSeek attention kernels require a zero-offset
                        // source allocation. Materialize the logical lane inside
                        // the same ordered command buffer rather than exposing a
                        // nonzero-offset view.
                        let sequence_state = if let Some(state) = state.as_ref() {
                            let sequence_state = alloc(
                                &device,
                                DType::F32,
                                vec![rows, hc, hidden],
                                "cooperative prefill sequence state",
                            )?;
                            dispatch_copy_f32(
                                session.encoder_mut(),
                                &mut self.ctx.registry,
                                device.metal_device(),
                                state,
                                &sequence_state,
                                element_offset,
                                0,
                                element_count,
                            )
                            .with_context(|| {
                                format!(
                                    "unpack DeepSeek-V4 cooperative prefill layer-{layer} sequence-{sequence} state"
                                )
                            })?;
                            session.barrier();
                            Some(sequence_state)
                        } else {
                            None
                        };
                        let attention = if layer == 0 {
                            anyhow::ensure!(
                                sequence_state.is_none(),
                                "DeepSeek-V4 cooperative layer 0 must embed each prompt"
                            );
                            self.forward_uncompressed_attention_prefill(
                                None,
                                token_batches[sequence],
                                layer,
                                caches[sequence],
                                &spans[sequence],
                                None,
                                Some(&mut session),
                            )
                        } else if self.cfg.compress_ratios[layer] == 0 {
                            self.forward_uncompressed_attention_prefill(
                                sequence_state.as_ref(),
                                token_batches[sequence],
                                layer,
                                caches[sequence],
                                &spans[sequence],
                                None,
                                Some(&mut session),
                            )
                        } else {
                            self.forward_compressed_attention_prefill(
                                sequence_state.as_ref().context(
                                    "DeepSeek-V4 cooperative compressed attention is missing state",
                                )?,
                                layer,
                                caches[sequence],
                                &spans[sequence],
                                None,
                                Some(&mut session),
                            )
                        }
                        .with_context(|| {
                            format!(
                                "encode DeepSeek-V4 cooperative prefill layer-{layer} sequence-{sequence} attention"
                            )
                        })?;
                        session.barrier();
                        dispatch_copy_f32(
                            session.encoder_mut(),
                            &mut self.ctx.registry,
                            device.metal_device(),
                            &attention,
                            &combined_attention,
                            0,
                            element_offset,
                            element_count,
                        )
                        .with_context(|| {
                            format!(
                                "pack DeepSeek-V4 cooperative prefill layer-{layer} sequence-{sequence} attention"
                            )
                        })?;
                        // Encoding errors above drop an uncommitted command
                        // buffer. Once finish begins, cache writes may have
                        // reached Metal even if completion reports an error.
                        submitted_any = true;
                        session.finish().with_context(|| {
                            format!(
                                "execute DeepSeek-V4 cooperative prefill layer-{layer} sequence-{sequence} attention"
                            )
                        })?;
                        Ok(())
                    })();
                    end_prefill_pool_layer();
                    attention_result?;
                    row_offset += rows;
                }

                begin_prefill_pool_layer();
                let ffn_result: Result<MlxBuffer> = (|| {
                    let mut session = executor.begin_recorded().with_context(|| {
                        format!("begin DeepSeek-V4 cooperative prefill layer {layer} FFN")
                    })?;
                    let next_state = self.forward_ffn_rows(
                        &combined_attention,
                        &combined_tokens,
                        layer,
                        None,
                        Some(&mut session),
                        Some(reusable_states[layer % reusable_states.len()].clone()),
                        id_mm_scratch.as_mut(),
                    )?;
                    session.finish().with_context(|| {
                        format!("execute DeepSeek-V4 cooperative prefill layer-{layer} FFN")
                    })?;
                    Ok(next_state)
                })();
                end_prefill_pool_layer();
                state = Some(ffn_result?);
            }
            let state = state.context("DeepSeek-V4 cooperative prefill encoded zero layers")?;
            let mut outputs = Vec::with_capacity(rows_per_sequence.len());
            let mut row_offset = 0_usize;
            for rows in rows_per_sequence {
                let element_offset = row_offset
                    .checked_mul(row_elements)
                    .context("DeepSeek-V4 cooperative output offset overflow")?;
                let element_count = rows
                    .checked_mul(row_elements)
                    .context("DeepSeek-V4 cooperative output size overflow")?;
                outputs.push(
                    state
                        .slice_view(
                            (element_offset * DType::F32.size_of()) as u64,
                            element_count,
                        )
                        .with_shape(vec![rows, hc, hidden])
                        .context("shape DeepSeek-V4 cooperative output view")?,
                );
                row_offset += rows;
            }
            anyhow::ensure!(
                row_offset == total_rows && combined_elements == state.element_count(),
                "DeepSeek-V4 cooperative prefill output partition drift"
            );
            Ok(outputs)
        })();
        (result, submitted_any)
    }

    fn forward_verifier_prefill_uncommitted(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
        span: &CacheSpan,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let executor = GraphExecutor::new(self.ctx.device().clone());
        let profile_timing = std::env::var_os("HF2Q_DEEPSEEK_PREFILL_TIMING").is_some();
        let graph_diag = std::env::var("HF2Q_DEEPSEEK_GRAPH_DIAG").as_deref() == Ok("1");
        let dump_intermediates = std::env::var_os("HF2Q_DEEPSEEK_DUMP_LAYER_DIR").is_some()
            || std::env::var_os("HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR").is_some();
        let graph_reorder = graph_reorder_enabled()?;
        let graph_layers_per_command_buffer = graph_layers_per_command_buffer(
            layers,
            graph_reorder,
            profile_timing || dump_intermediates,
        )?;
        if graph_layers_per_command_buffer > 1 {
            anyhow::ensure!(
                graph_reorder,
                "multi-layer DeepSeek-V4 graphs require HF2Q_DEEPSEEK_GRAPH_REORDER=1"
            );
            anyhow::ensure!(
                retained_reference_pipeline_enabled(),
                "multi-layer DeepSeek-V4 graphs require retained Metal command-buffer references"
            );
            anyhow::ensure!(
                !profile_timing,
                "multi-layer DeepSeek-V4 graphs are incompatible with per-layer timing"
            );
            anyhow::ensure!(
                !dump_intermediates,
                "multi-layer DeepSeek-V4 graphs are incompatible with intermediate state dumps"
            );
        }
        let device = self.ctx.device().clone();
        let rows = token_ids.len();
        let hc = self.cfg.hyper_connection_count as usize;
        let hidden = self.cfg.hidden_size as usize;
        let mut id_mm_scratch = if rows > MM_ID_ROUTING_THRESHOLD as usize {
            let rows = u32::try_from(rows).context("DeepSeek-V4 prefill rows exceed u32")?;
            Some([
                IdMmScratch::alloc(self.ctx.device(), self.cfg.num_experts, rows)
                    .context("allocate DeepSeek-V4 gate/down mm_id scratch")?,
                IdMmScratch::alloc(self.ctx.device(), self.cfg.num_experts, rows)
                    .context("allocate DeepSeek-V4 up mm_id scratch")?,
            ])
        } else {
            None
        };
        // Each layer fully overwrites its output after reading the preceding
        // layer's state. Alternating two persistent buffers preserves that
        // dependency while avoiding 43 fresh, CPU-zeroed Metal allocations
        // for every prompt chunk.
        let reusable_states = [
            alloc_persistent(
                &device,
                DType::F32,
                vec![rows, hc, hidden],
                "prefill state ping",
            )?,
            alloc_persistent(
                &device,
                DType::F32,
                vec![rows, hc, hidden],
                "prefill state pong",
            )?,
        ];
        let mut state = None;
        if graph_layers_per_command_buffer > 1 {
            for start in (0..layers).step_by(graph_layers_per_command_buffer) {
                let end = (start + graph_layers_per_command_buffer).min(layers);
                begin_prefill_submission_inputs();
                let group_result: Result<()> = (|| {
                    let mut session = executor.begin_recorded().with_context(|| {
                        format!("begin DeepSeek-V4 recorded prefill layers {start}..{end}")
                    })?;
                    for layer in start..end {
                        begin_prefill_pool_layer();
                        let layer_result = self.encode_verifier_layer_prefill(
                            token_ids,
                            layer,
                            state.as_ref(),
                            cache,
                            span,
                            reusable_states[layer % reusable_states.len()].clone(),
                            id_mm_scratch.as_mut(),
                            &mut session,
                        );
                        if layer_result.is_ok() && layer + 1 < end {
                            session.barrier();
                        }
                        end_prefill_pool_layer();
                        state = Some(layer_result?);
                    }
                    let (reordered, barriers) =
                        session.finish_with_reorder().with_context(|| {
                            format!("reorder DeepSeek-V4 prefill layers {start}..{end}")
                        })?;
                    if graph_diag {
                        eprintln!(
                            "[GRAPH_REORDER] layers={start}..{end} reordered={reordered} barriers={barriers}"
                        );
                    }
                    Ok(())
                })();
                end_prefill_submission_inputs();
                if let Err(error) = &group_result {
                    eprintln!("[GRAPH_REORDER] layers={start}..{end} failed: {error:#}");
                }
                group_result?;
            }
            return state.context("DeepSeek-V4 multi-layer graph encoded zero layers");
        }
        for layer in 0..layers {
            begin_prefill_pool_layer();
            let layer_start = std::time::Instant::now();
            let layer_result: Result<MlxBuffer> = (|| {
                let record_graph = (layer == 0 && graph_diag) || graph_reorder;
                let mut session = if record_graph {
                    executor.begin_recorded()
                } else {
                    executor.begin()
                }
                .with_context(|| format!("begin DeepSeek-V4 prefill layer {layer}"))?;
                let next_state = self.encode_verifier_layer_prefill(
                    token_ids,
                    layer,
                    state.as_ref(),
                    cache,
                    span,
                    reusable_states[layer % reusable_states.len()].clone(),
                    id_mm_scratch.as_mut(),
                    &mut session,
                )?;
                if graph_reorder {
                    let (reordered, barriers) = session
                        .finish_with_reorder()
                        .with_context(|| format!("reorder DeepSeek-V4 prefill layer {layer}"))?;
                    if graph_diag {
                        eprintln!(
                            "[GRAPH_REORDER] layer={layer} reordered={reordered} barriers={barriers}"
                        );
                    }
                } else if profile_timing {
                    let (encode_ns, wait_ns) = session
                        .finish_with_timing(layer_start)
                        .with_context(|| format!("execute DeepSeek-V4 prefill layer {layer}"))?;
                    eprintln!(
                        "DeepSeek-V4 prefill layer {layer}: encode {:.3} ms; commit/wait {:.3} ms",
                        encode_ns as f64 / 1e6,
                        wait_ns as f64 / 1e6
                    );
                } else {
                    session
                        .finish()
                        .with_context(|| format!("execute DeepSeek-V4 prefill layer {layer}"))?;
                }
                self.dump_verifier_layer_state(
                    &next_state,
                    layer,
                    span.start_position + span.token_count,
                )?;
                Ok(next_state)
            })();
            if graph_reorder {
                if let Err(error) = &layer_result {
                    eprintln!("[GRAPH_REORDER] layer={layer} failed: {error:#}");
                }
            }
            let reset_start = std::time::Instant::now();
            end_prefill_pool_layer();
            if profile_timing {
                eprintln!(
                    "DeepSeek-V4 prefill layer {layer}: pool reset {:.3} ms; total {:.3} ms",
                    reset_start.elapsed().as_secs_f64() * 1e3,
                    layer_start.elapsed().as_secs_f64() * 1e3
                );
            }
            state = Some(layer_result?);
        }
        state.context("DeepSeek-V4 prefill encoded zero layers")
    }

    fn encode_verifier_layer_prefill(
        &mut self,
        token_ids: &[u32],
        layer: usize,
        state: Option<&MlxBuffer>,
        cache: &mut Deepseek4Cache,
        span: &CacheSpan,
        output_state: MlxBuffer,
        id_mm_scratch: Option<&mut [IdMmScratch; 2]>,
        session: &mut GraphSession<'_>,
    ) -> Result<MlxBuffer> {
        let dump_attention = std::env::var_os("HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR").is_some();
        let attention_session = (!dump_attention).then_some(&mut *session);
        let attention = if layer == 0 {
            anyhow::ensure!(state.is_none(), "DeepSeek-V4 layer 0 must embed the prompt");
            self.forward_uncompressed_attention_prefill(
                None,
                token_ids,
                layer,
                cache,
                span,
                None,
                attention_session,
            )
        } else {
            let state = state.context("DeepSeek-V4 nonzero prefill layer is missing state")?;
            if self.cfg.compress_ratios[layer] == 0 {
                self.forward_uncompressed_attention_prefill(
                    Some(state),
                    token_ids,
                    layer,
                    cache,
                    span,
                    None,
                    attention_session,
                )
            } else {
                self.forward_compressed_attention_prefill(
                    state,
                    layer,
                    cache,
                    span,
                    None,
                    attention_session,
                )
            }
        }
        .with_context(|| format!("encode DeepSeek-V4 prefill layer-{layer} attention"))?;
        self.dump_verifier_attention_state(
            &attention,
            layer,
            span.start_position + span.token_count,
        )?;
        if std::env::var("HF2Q_DEEPSEEK_ENCODER_STAGES").as_deref() == Ok("1") {
            session
                .encoder_mut()
                .profile_stage_boundary(match self.cfg.compress_ratios[layer] {
                    0 => "DeepSeek-V4 uncompressed attention",
                    4 => "DeepSeek-V4 compressed attention ratio-4",
                    128 => "DeepSeek-V4 compressed attention ratio-128",
                    _ => "DeepSeek-V4 compressed attention unknown-ratio",
                })
                .with_context(|| format!("profile DeepSeek-V4 layer-{layer} attention"))?;
        }
        self.forward_ffn_rows(
            &attention,
            token_ids,
            layer,
            None,
            Some(session),
            Some(output_state),
            id_mm_scratch,
        )
        .with_context(|| format!("encode DeepSeek-V4 prefill layer-{layer} FFN"))
    }

    /// Execute all verifier blocks for one token.
    ///
    /// The normal retained-resource path encodes several complete layers per
    /// command buffer, preserving CPU/GPU overlap without the original
    /// attention/FFN submission fragmentation. Diagnostic stage profiling and
    /// unretained-resource mode keep the isolated stage path. Any submitted GPU
    /// failure poisons the cache; the caller must reset and replay rather than
    /// retry only the failed token.
    pub fn forward_verifier_one(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        self.forward_verifier_one_with_commit_gate(token_id, cache, || Ok(()))
    }

    /// One-token verifier sibling of
    /// [`Self::forward_verifier_prefill_with_commit_gate`]. The graph may
    /// complete, but a late supervisor verdict poisons the cache without
    /// advancing its logical cursor.
    pub fn forward_verifier_one_with_commit_gate(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
        commit_gate: impl FnOnce() -> Result<()>,
    ) -> Result<MlxBuffer> {
        let position = cache.position();
        let result = self.forward_verifier_one_uncommitted(token_id, cache);
        match result {
            Ok(state) => {
                publish_verifier_cache_after_gate(cache, position, commit_gate)?;
                Ok(state)
            }
            Err(error) => {
                cache.poison();
                Err(error).context("DeepSeek-V4 verifier token partially executed; cache poisoned")
            }
        }
    }

    fn forward_verifier_one_uncommitted(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let retained = retained_reference_pipeline_enabled();
        let profile_stages = std::env::var("HF2Q_DEEPSEEK_STAGE_PROFILE").as_deref() == Ok("1");
        let capture_stage_profile = profile_stages
            && STAGE_PROFILE_CAPTURED
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
        if capture_stage_profile {
            mlx_native::kernel_profile::reset();
        }
        let dump_layers = std::env::var_os("HF2Q_DEEPSEEK_DUMP_LAYER_DIR").is_some()
            || std::env::var_os("HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR").is_some();
        begin_decode_pool_token();
        let token_result = if retained && !capture_stage_profile && !dump_layers {
            self.forward_verifier_one_chunked(token_id, cache)
        } else {
            let pipelined = retained && !dump_layers;
            let mut in_flight = SubmissionChain::with_capacity(layers.saturating_mul(2));
            let result = self.encode_verifier_layers(
                token_id,
                cache,
                None,
                pipelined.then_some(&mut in_flight),
            );
            let drained = drain(&in_flight).context("drain DeepSeek-V4 verifier pipeline");
            if capture_stage_profile {
                let profile = mlx_native::kernel_profile::dump();
                let total_ns: u64 = profile.iter().map(|(_, entry)| entry.total_ns).sum();
                let attention_ns: u64 = profile
                    .iter()
                    .filter(|(label, _)| label.contains("attention"))
                    .map(|(_, entry)| entry.total_ns)
                    .sum();
                let ffn_ns: u64 = profile
                    .iter()
                    .filter(|(label, _)| label.contains("FFN"))
                    .map(|(_, entry)| entry.total_ns)
                    .sum();
                eprintln!(
                    "DeepSeek-V4 one-token GPU stage profile: total {:.3} ms; attention {:.3} ms; FFN {:.3} ms; command_buffers={}",
                    total_ns as f64 / 1e6,
                    attention_ns as f64 / 1e6,
                    ffn_ns as f64 / 1e6,
                    profile.len(),
                );
                for (label, entry) in profile.iter().take(12) {
                    eprintln!(
                        "DeepSeek-V4 stage {label}: {:.3} ms",
                        entry.total_ns as f64 / 1e6
                    );
                }
            }
            drop(in_flight);
            match (result, drained) {
                (Ok(state), Ok(())) => Ok(state),
                (Err(error), Ok(())) => Err(error),
                (Ok(_), Err(error)) => Err(error),
                (Err(error), Err(drain_error)) => {
                    Err(error).context(format!("pipeline drain also failed: {drain_error:#}"))
                }
            }
        };
        end_decode_pool_token();
        token_result
    }

    fn forward_verifier_one_chunked(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let layers_per_command_buffer = std::env::var("HF2Q_DEEPSEEK_LAYERS_PER_CB")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .context("HF2Q_DEEPSEEK_LAYERS_PER_CB must be a positive integer")
            })
            .transpose()?
            .unwrap_or(2);
        anyhow::ensure!(
            (1..=layers).contains(&layers_per_command_buffer),
            "HF2Q_DEEPSEEK_LAYERS_PER_CB must be in 1..={layers}"
        );

        let executor = GraphExecutor::new(self.ctx.device().clone());
        let command_buffers = layers.div_ceil(layers_per_command_buffer);
        let mut in_flight = SubmissionChain::with_capacity(command_buffers);
        let result = (|| {
            let mut state = None;
            for start in (0..layers).step_by(layers_per_command_buffer) {
                let end = (start + layers_per_command_buffer).min(layers);
                let mut session = executor
                    .begin()
                    .with_context(|| format!("begin DeepSeek-V4 verifier layers {start}..{end}"))?;
                for layer in start..end {
                    state = Some(self.encode_verifier_layer(
                        token_id,
                        layer,
                        state.as_ref(),
                        cache,
                        Some(&mut session),
                        None,
                    )?);
                }
                in_flight.push((
                    format!("execute DeepSeek-V4 verifier layers {start}..{end}"),
                    session.commit(),
                ));
            }
            state.context("DeepSeek-V4 verifier encoded zero layers")
        })();
        let drained = drain(&in_flight).context("drain chunked DeepSeek-V4 verifier pipeline");
        drop(in_flight);
        match (result, drained) {
            (Ok(state), Ok(())) => Ok(state),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(drain_error)) => Err(error).context(format!(
                "chunked pipeline drain also failed: {drain_error:#}"
            )),
        }
    }

    fn encode_verifier_layers(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
        mut shared_session: Option<&mut GraphSession<'_>>,
        mut in_flight: Option<&mut SubmissionChain>,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let mut state = None;
        for layer in 0..layers {
            let next_state = self.encode_verifier_layer(
                token_id,
                layer,
                state.as_ref(),
                cache,
                shared_session.as_deref_mut(),
                in_flight.as_deref_mut(),
            )?;
            self.dump_verifier_layer_state(&next_state, layer, cache.position() + 1)?;
            state = Some(next_state);
        }
        state.context("DeepSeek-V4 verifier encoded zero layers")
    }

    fn dump_verifier_layer_state(
        &self,
        state: &MlxBuffer,
        layer: usize,
        position: usize,
    ) -> Result<()> {
        let Some(directory) = std::env::var_os("HF2Q_DEEPSEEK_DUMP_LAYER_DIR") else {
            return Ok(());
        };
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).with_context(|| {
            format!(
                "create DeepSeek-V4 layer dump directory {}",
                directory.display()
            )
        })?;
        let last = self
            .last_token_state(state)
            .context("view DeepSeek-V4 diagnostic layer state")?;
        let elements = self.cfg.hyper_connection_count as usize * self.cfg.hidden_size as usize;
        crate::debug::dumps::dump_f32_to(
            &last,
            elements,
            "deepseek_layer_state",
            Some(layer),
            position,
            Some(&directory),
        )
    }

    fn dump_verifier_attention_state(
        &self,
        state: &MlxBuffer,
        layer: usize,
        position: usize,
    ) -> Result<()> {
        let Some(directory) = std::env::var_os("HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR") else {
            return Ok(());
        };
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).with_context(|| {
            format!(
                "create DeepSeek-V4 attention dump directory {}",
                directory.display()
            )
        })?;
        let last = self
            .last_token_state(state)
            .context("view DeepSeek-V4 diagnostic attention state")?;
        let elements = self.cfg.hyper_connection_count as usize * self.cfg.hidden_size as usize;
        crate::debug::dumps::dump_f32_to(
            &last,
            elements,
            "deepseek_attention_state",
            Some(layer),
            position,
            Some(&directory),
        )
    }

    fn encode_verifier_layer(
        &mut self,
        token_id: u32,
        layer: usize,
        state: Option<&MlxBuffer>,
        cache: &mut Deepseek4Cache,
        shared_session: Option<&mut GraphSession<'_>>,
        in_flight: Option<&mut SubmissionChain>,
    ) -> Result<MlxBuffer> {
        let mut shared_session = shared_session;
        let mut in_flight = in_flight;
        let attention = if layer == 0 {
            anyhow::ensure!(
                state.is_none(),
                "DeepSeek-V4 layer 0 must embed the input token"
            );
            self.forward_uncompressed_attention_one(
                None,
                token_id,
                layer,
                cache,
                false,
                in_flight.as_deref_mut(),
                shared_session.as_deref_mut(),
            )
        } else {
            let state = state.context("DeepSeek-V4 nonzero layer is missing its input state")?;
            if self.cfg.compress_ratios[layer] == 0 {
                self.forward_uncompressed_attention_one(
                    Some(state),
                    token_id,
                    layer,
                    cache,
                    false,
                    in_flight.as_deref_mut(),
                    shared_session.as_deref_mut(),
                )
            } else {
                self.forward_compressed_attention_one(
                    state,
                    layer,
                    cache,
                    false,
                    in_flight.as_deref_mut(),
                    shared_session.as_deref_mut(),
                )
            }
        }
        .with_context(|| format!("execute DeepSeek-V4 layer-{layer} attention"))?;
        self.dump_verifier_attention_state(&attention, layer, cache.position() + 1)?;
        self.forward_ffn_one(
            &attention,
            token_id,
            layer,
            in_flight.as_deref_mut(),
            shared_session.as_deref_mut(),
        )
        .with_context(|| format!("execute DeepSeek-V4 layer-{layer} FFN"))
    }
}

#[cfg(test)]
mod prompt_chunk_tests {
    use super::{
        default_graph_layers_per_command_buffer, matrix_prefill_chunk_len,
        prefill_windows_for_resident_bytes, DEFAULT_MATRIX_PREFILL_WINDOWS,
        LARGE_MODEL_MATRIX_PREFILL_WINDOWS, LARGE_MODEL_RESIDENT_BYTES, MIN_MATRIX_APPEND_TOKENS,
    };

    #[test]
    fn reordered_prefill_defaults_to_four_layers_without_breaking_diagnostics() {
        assert_eq!(default_graph_layers_per_command_buffer(43, true, false), 4);
        assert_eq!(default_graph_layers_per_command_buffer(2, true, false), 2);
        assert_eq!(default_graph_layers_per_command_buffer(43, false, false), 1);
        assert_eq!(default_graph_layers_per_command_buffer(43, true, true), 1);
    }

    #[test]
    fn long_prompts_continue_matrix_prefill_after_the_first_chunk() {
        assert_eq!(matrix_prefill_chunk_len(0, 6_000, 128, 16), 2_048);
        assert_eq!(matrix_prefill_chunk_len(2_048, 3_952, 128, 16), 2_048);
        assert_eq!(matrix_prefill_chunk_len(0, 6_000, 128, 32), 4_096);
        assert_eq!(
            matrix_prefill_chunk_len(0, 6_000, 128, 2),
            256,
            "interactive mixed work caps a transaction without changing the solo plan"
        );
    }

    #[test]
    fn only_large_resident_artifacts_use_the_2k_transaction() {
        assert_eq!(
            prefill_windows_for_resident_bytes(LARGE_MODEL_RESIDENT_BYTES - 1),
            DEFAULT_MATRIX_PREFILL_WINDOWS
        );
        assert_eq!(
            prefill_windows_for_resident_bytes(LARGE_MODEL_RESIDENT_BYTES),
            LARGE_MODEL_MATRIX_PREFILL_WINDOWS
        );
    }

    #[test]
    fn only_small_cached_suffixes_use_incremental_replay() {
        assert_eq!(
            matrix_prefill_chunk_len(1_024, MIN_MATRIX_APPEND_TOKENS - 1, 128, 16),
            0
        );
        assert_eq!(
            matrix_prefill_chunk_len(1_024, MIN_MATRIX_APPEND_TOKENS, 128, 16),
            MIN_MATRIX_APPEND_TOKENS
        );
    }
}
