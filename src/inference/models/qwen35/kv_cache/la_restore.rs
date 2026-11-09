use anyhow::{anyhow, Context, Result};
use mlx_native::ops::copy::{dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer};

use crate::serve::multi_seq_kv::SlotId;

use super::{HybridKvCache, LinearAttnStateSlot};
use crate::inference::models::qwen35::Qwen35Variant;

/// DeltaNet capture-to-state restore implementation selected for a cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaRestoreRoute {
    Host,
    Gpu,
}

impl LaRestoreRoute {
    pub(crate) fn resolve(value: Option<&str>, variant: Qwen35Variant) -> Result<Self> {
        match value.map(str::trim) {
            Some("0") => Ok(Self::Host),
            Some("1") => Ok(Self::Gpu),
            Some(value) => Err(anyhow!(
                "HF2Q_QWEN_LA_GPU_RESTORE must be 0 or 1 (got {value:?})"
            )),
            None => match variant {
                // The matched P4b A/B was a Win with every promotion gate
                // passing. Qwen3.6 remains host-only without an override.
                Qwen35Variant::Dense => Ok(Self::Gpu),
                Qwen35Variant::Moe => Ok(Self::Host),
            },
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaRestoreItem {
    pub slot: SlotId,
    pub capture_row: u32,
}

#[derive(Clone)]
pub(crate) struct LaAnchorImage {
    storage: MlxBuffer,
    layers: Vec<LaAnchorLayer>,
}

#[derive(Clone, Copy)]
struct LaAnchorLayer {
    recurrent_offset: usize,
    recurrent_elements: usize,
    conv_offset: usize,
    conv_elements: usize,
    channels: usize,
    k_minus_one: usize,
}

impl LaAnchorImage {
    pub(crate) fn total_bytes(&self) -> usize {
        self.storage.data_byte_len()
    }

    pub(crate) fn is_right_sized(&self) -> bool {
        self.storage.byte_len() == self.storage.data_byte_len()
    }
}

pub(crate) struct LaAnchorRestorePlan {
    slot: SlotId,
    cursors: Vec<u32>,
    mtp_cursor: Option<u32>,
    copies: LaRestorePlan,
}

struct ConvCopy {
    source: MlxBuffer,
    destination: MlxBuffer,
    channels: u32,
    k_minus_one: u32,
}

struct RecurrentCopy {
    source: MlxBuffer,
    destination: MlxBuffer,
    bytes: u64,
}

struct RestoreCopy {
    conv: ConvCopy,
    recurrent: RecurrentCopy,
}

struct SnapshotCopy {
    conv_source: MlxBuffer,
    conv_destination: MlxBuffer,
    recurrent_source: MlxBuffer,
    recurrent_destination: MlxBuffer,
    recurrent_bytes: u64,
    channels: u32,
    k_minus_one: u32,
}

/// Fully validated capture-to-state restore. Every view is bounded before an
/// encoder is created, so a refusal cannot partially mutate GPU state.
pub(crate) struct LaRestorePlan {
    copies: Vec<RestoreCopy>,
}

fn current_recurrent(layer: &LinearAttnStateSlot, slot: usize) -> &MlxBuffer {
    if layer.pp_flipped[slot] {
        &layer.recurrent_scratch
    } else {
        &layer.recurrent
    }
}

fn current_conv(layer: &LinearAttnStateSlot, slot: usize) -> &MlxBuffer {
    if layer.pp_flipped[slot] {
        &layer.conv_state_scratch
    } else {
        &layer.conv_state
    }
}

fn f32_view(
    buffer: &MlxBuffer,
    offset_elements: usize,
    elements: usize,
    what: &str,
) -> Result<MlxBuffer> {
    if buffer.dtype() != DType::F32 {
        return Err(anyhow!("{what}: expected F32, got {:?}", buffer.dtype()));
    }
    let offset_bytes = offset_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("{what}: byte offset overflow"))?;
    let byte_len = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("{what}: byte length overflow"))?;
    let end = offset_bytes
        .checked_add(byte_len)
        .ok_or_else(|| anyhow!("{what}: byte range overflow"))?;
    if end > buffer.data_byte_len() {
        return Err(anyhow!(
            "{what}: byte range [{offset_bytes}, {end}) exceeds logical length {}",
            buffer.data_byte_len()
        ));
    }
    let relative_offset =
        u64::try_from(offset_bytes).context("f32_view: offset does not fit u64")?;
    buffer
        .byte_offset()
        .checked_add(relative_offset)
        .ok_or_else(|| anyhow!("{what}: absolute byte offset overflow"))?;
    Ok(buffer.slice_view(relative_offset, elements))
}

fn views_overlap(left: &MlxBuffer, right: &MlxBuffer) -> Result<bool> {
    if left.contents_ptr() != right.contents_ptr() {
        return Ok(false);
    }
    let left_end = left
        .byte_offset()
        .checked_add(
            u64::try_from(left.data_byte_len())
                .context("views_overlap: left length does not fit u64")?,
        )
        .ok_or_else(|| anyhow!("views_overlap: left range overflow"))?;
    let right_end = right
        .byte_offset()
        .checked_add(
            u64::try_from(right.data_byte_len())
                .context("views_overlap: right length does not fit u64")?,
        )
        .ok_or_else(|| anyhow!("views_overlap: right range overflow"))?;
    Ok(left.byte_offset() < right_end && right.byte_offset() < left_end)
}

fn anchor_layout(cache: &HybridKvCache) -> Result<(Vec<LaAnchorLayer>, usize)> {
    let n_seqs = cache.n_seqs as usize;
    anyhow::ensure!(n_seqs > 0, "anchor image: n_seqs is zero");
    let mut offset = 0usize;
    let mut layers = Vec::with_capacity(cache.linear_attn.len());
    for (index, layer) in cache.linear_attn.iter().enumerate() {
        anyhow::ensure!(
            layer.recurrent.element_count() % n_seqs == 0,
            "anchor image: recurrent layer {index} is not slot-aligned"
        );
        let recurrent_elements = layer.recurrent.element_count() / n_seqs;
        let shape = layer.conv_state.shape();
        anyhow::ensure!(
            shape.len() == 3 && shape[2] == n_seqs,
            "anchor image: conv layer {index} shape {shape:?} is not [channels, K-1, n_seqs]"
        );
        let channels = shape[0];
        let k_minus_one = shape[1];
        let conv_elements = channels
            .checked_mul(k_minus_one)
            .ok_or_else(|| anyhow!("anchor image: conv element overflow"))?;
        let recurrent_offset = offset;
        offset = offset
            .checked_add(recurrent_elements)
            .ok_or_else(|| anyhow!("anchor image: recurrent offset overflow"))?;
        let conv_offset = offset;
        offset = offset
            .checked_add(conv_elements)
            .ok_or_else(|| anyhow!("anchor image: conv offset overflow"))?;
        layers.push(LaAnchorLayer {
            recurrent_offset,
            recurrent_elements,
            conv_offset,
            conv_elements,
            channels,
            k_minus_one,
        });
    }
    Ok((layers, offset))
}

impl HybridKvCache {
    pub(crate) fn set_la_restore_route(&mut self, route: LaRestoreRoute) {
        self.la_restore_route = route;
    }

    #[cfg(test)]
    pub(crate) fn la_restore_route(&self) -> LaRestoreRoute {
        self.la_restore_route
    }

    pub(crate) fn slot_anchor_image_bytes(&self) -> Result<usize> {
        let (_, elements) = anchor_layout(self)?;
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("anchor image byte length overflow"))
    }

    pub(crate) fn snapshot_la_anchor(&mut self, slot: SlotId) -> Result<LaAnchorImage> {
        let slot_index = slot.0 as usize;
        let n_seqs = self.n_seqs as usize;
        anyhow::ensure!(
            slot_index < n_seqs,
            "snapshot_la_anchor: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        let (layers, elements) = anchor_layout(self)?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("snapshot_la_anchor: image byte length overflow"))?;
        let mut image = LaAnchorImage {
            storage: self
                .device
                .alloc_buffer(bytes, DType::F32, vec![elements])
                .map_err(|error| anyhow!("snapshot_la_anchor: allocate image: {error}"))?,
            layers,
        };
        anyhow::ensure!(
            image.is_right_sized(),
            "snapshot_la_anchor: image owner is not right-sized"
        );

        match self.la_restore_route {
            LaRestoreRoute::Host => {
                let destination = image
                    .storage
                    .as_mut_slice::<u32>()
                    .context("snapshot_la_anchor: map image")?;
                for (layer, layout) in self.linear_attn.iter().zip(&image.layers) {
                    let recurrent_per_slot = layout.recurrent_elements;
                    let recurrent_start = slot_index
                        .checked_mul(recurrent_per_slot)
                        .ok_or_else(|| anyhow!("snapshot_la_anchor: recurrent slot overflow"))?;
                    let recurrent = current_recurrent(layer, slot_index)
                        .as_slice::<u32>()
                        .context("snapshot_la_anchor: map recurrent state")?;
                    destination
                        [layout.recurrent_offset..layout.recurrent_offset + recurrent_per_slot]
                        .copy_from_slice(
                            &recurrent[recurrent_start..recurrent_start + recurrent_per_slot],
                        );

                    let conv_start = slot_index
                        .checked_mul(layout.conv_elements)
                        .ok_or_else(|| anyhow!("snapshot_la_anchor: conv slot overflow"))?;
                    let conv = current_conv(layer, slot_index)
                        .as_slice::<u32>()
                        .context("snapshot_la_anchor: map conv state")?;
                    let output = &mut destination
                        [layout.conv_offset..layout.conv_offset + layout.conv_elements];
                    for k in 0..layout.k_minus_one {
                        for channel in 0..layout.channels {
                            output[k * layout.channels + channel] =
                                conv[conv_start + channel * layout.k_minus_one + k];
                        }
                    }
                }
            }
            LaRestoreRoute::Gpu => {
                let mut copies = Vec::with_capacity(self.linear_attn.len());
                for (layer, layout) in self.linear_attn.iter().zip(&image.layers) {
                    let recurrent_source = f32_view(
                        current_recurrent(layer, slot_index),
                        slot_index
                            .checked_mul(layout.recurrent_elements)
                            .ok_or_else(|| {
                                anyhow!("snapshot_la_anchor: recurrent source overflow")
                            })?,
                        layout.recurrent_elements,
                        "snapshot_la_anchor recurrent source",
                    )?;
                    let recurrent_destination = f32_view(
                        &image.storage,
                        layout.recurrent_offset,
                        layout.recurrent_elements,
                        "snapshot_la_anchor recurrent destination",
                    )?;
                    let conv_source = f32_view(
                        current_conv(layer, slot_index),
                        slot_index
                            .checked_mul(layout.conv_elements)
                            .ok_or_else(|| anyhow!("snapshot_la_anchor: conv source overflow"))?,
                        layout.conv_elements,
                        "snapshot_la_anchor conv source",
                    )?;
                    let conv_destination = f32_view(
                        &image.storage,
                        layout.conv_offset,
                        layout.conv_elements,
                        "snapshot_la_anchor conv destination",
                    )?;
                    copies.push(SnapshotCopy {
                        conv_source,
                        conv_destination,
                        recurrent_source,
                        recurrent_destination,
                        recurrent_bytes: u64::try_from(
                            layout.recurrent_elements * std::mem::size_of::<f32>(),
                        )
                        .context("snapshot_la_anchor: recurrent bytes do not fit u64")?,
                        channels: u32::try_from(layout.channels)
                            .context("snapshot_la_anchor: channels do not fit u32")?,
                        k_minus_one: u32::try_from(layout.k_minus_one)
                            .context("snapshot_la_anchor: K-1 does not fit u32")?,
                    });
                }
                let mut encoder = self
                    .device
                    .command_encoder()
                    .context("snapshot_la_anchor: create command encoder")?;
                let registry = self
                    .la_restore_registry
                    .get_or_insert_with(KernelRegistry::new);
                for copy in &copies {
                    dispatch_strided_copy_f32(
                        &mut encoder,
                        registry,
                        self.device.metal_device(),
                        &copy.conv_source,
                        &copy.conv_destination,
                        &StridedCopyParams {
                            rows: copy.k_minus_one,
                            cols: copy.channels,
                            stride_row: 1,
                            stride_col: copy.k_minus_one,
                        },
                    )
                    .context("snapshot_la_anchor: conv strided copy")?;
                }
                for copy in &copies {
                    encoder
                        .blit_copy_bytes(
                            &copy.recurrent_source,
                            0,
                            &copy.recurrent_destination,
                            0,
                            copy.recurrent_bytes,
                        )
                        .context("snapshot_la_anchor: recurrent blit")?;
                }
                encoder
                    .commit_and_wait_labeled("qwen35.anchor_snapshot")
                    .context("snapshot_la_anchor: GPU snapshot")?;
            }
        }
        Ok(image)
    }

    pub(crate) fn plan_anchor_restore(
        &self,
        slot: SlotId,
        image: &LaAnchorImage,
        cursors: &[u32],
        mtp_cursor: Option<u32>,
    ) -> Result<LaAnchorRestorePlan> {
        let slot_index = slot.0 as usize;
        anyhow::ensure!(
            slot_index < self.n_seqs as usize,
            "plan_anchor_restore: slot {} outside n_seqs={}",
            slot.0,
            self.n_seqs
        );
        anyhow::ensure!(
            cursors.len() == self.full_attn.len(),
            "plan_anchor_restore: full-attention layer count mismatch"
        );
        anyhow::ensure!(
            image.layers.len() == self.linear_attn.len(),
            "plan_anchor_restore: linear-attention layer count mismatch"
        );
        let (expected_layers, expected_elements) = anchor_layout(self)?;
        anyhow::ensure!(
            image.storage.element_count() == expected_elements
                && image
                    .layers
                    .iter()
                    .zip(&expected_layers)
                    .all(|(actual, expected)| {
                        actual.recurrent_offset == expected.recurrent_offset
                            && actual.recurrent_elements == expected.recurrent_elements
                            && actual.conv_offset == expected.conv_offset
                            && actual.conv_elements == expected.conv_elements
                            && actual.channels == expected.channels
                            && actual.k_minus_one == expected.k_minus_one
                    }),
            "plan_anchor_restore: anchor image geometry mismatch"
        );
        for (index, (layer, &saved)) in self.full_attn.iter().zip(cursors).enumerate() {
            let live = *layer
                .current_len
                .get(slot_index)
                .ok_or_else(|| anyhow!("plan_anchor_restore: cursor missing at layer {index}"))?;
            anyhow::ensure!(
                live >= saved,
                "plan_anchor_restore: live cursor {live} behind saved cursor {saved} at layer {index}"
            );
        }
        match (&self.mtp_slot, mtp_cursor) {
            (Some(mtp), Some(saved)) => {
                let live = *mtp.current_len.get(slot_index).ok_or_else(|| {
                    anyhow!(
                        "plan_anchor_restore: MTP cursor missing for slot {}",
                        slot.0
                    )
                })?;
                anyhow::ensure!(
                    live >= saved,
                    "plan_anchor_restore: live MTP cursor {live} behind saved {saved}"
                );
            }
            (None, None) => {}
            _ => anyhow::bail!("plan_anchor_restore: MTP presence mismatch"),
        }

        let mut copies = Vec::with_capacity(self.linear_attn.len());
        for (layer, layout) in self.linear_attn.iter().zip(&image.layers) {
            let recurrent_source = f32_view(
                &image.storage,
                layout.recurrent_offset,
                layout.recurrent_elements,
                "plan_anchor_restore recurrent source",
            )?;
            let recurrent_destination = f32_view(
                &layer.recurrent,
                slot_index
                    .checked_mul(layout.recurrent_elements)
                    .ok_or_else(|| {
                        anyhow!("plan_anchor_restore: recurrent destination overflow")
                    })?,
                layout.recurrent_elements,
                "plan_anchor_restore recurrent destination",
            )?;
            let conv_source = f32_view(
                &image.storage,
                layout.conv_offset,
                layout.conv_elements,
                "plan_anchor_restore conv source",
            )?;
            let conv_destination = f32_view(
                &layer.conv_state,
                slot_index
                    .checked_mul(layout.conv_elements)
                    .ok_or_else(|| anyhow!("plan_anchor_restore: conv destination overflow"))?,
                layout.conv_elements,
                "plan_anchor_restore conv destination",
            )?;
            anyhow::ensure!(
                !views_overlap(&recurrent_source, &recurrent_destination)?
                    && !views_overlap(&conv_source, &conv_destination)?,
                "plan_anchor_restore: source overlaps destination"
            );
            copies.push(RestoreCopy {
                conv: ConvCopy {
                    source: conv_source,
                    destination: conv_destination,
                    channels: u32::try_from(layout.channels)
                        .context("plan_anchor_restore: channels do not fit u32")?,
                    k_minus_one: u32::try_from(layout.k_minus_one)
                        .context("plan_anchor_restore: K-1 does not fit u32")?,
                },
                recurrent: RecurrentCopy {
                    source: recurrent_source,
                    destination: recurrent_destination,
                    bytes: u64::try_from(layout.recurrent_elements * std::mem::size_of::<f32>())
                        .context("plan_anchor_restore: recurrent bytes do not fit u64")?,
                },
            });
        }
        Ok(LaAnchorRestorePlan {
            slot,
            cursors: cursors.to_vec(),
            mtp_cursor,
            copies: LaRestorePlan { copies },
        })
    }

    pub(crate) fn apply_anchor_restore(&mut self, mut plan: LaAnchorRestorePlan) -> Result<()> {
        match self.la_restore_route {
            LaRestoreRoute::Host => {
                for copy in &mut plan.copies.copies {
                    let source = copy
                        .conv
                        .source
                        .as_slice::<u32>()
                        .context("apply_anchor_restore: map conv source")?;
                    let destination = copy
                        .conv
                        .destination
                        .as_mut_slice::<u32>()
                        .context("apply_anchor_restore: map conv destination")?;
                    let channels = copy.conv.channels as usize;
                    let k_minus_one = copy.conv.k_minus_one as usize;
                    for channel in 0..channels {
                        for k in 0..k_minus_one {
                            destination[channel * k_minus_one + k] = source[k * channels + channel];
                        }
                    }
                    let source = copy
                        .recurrent
                        .source
                        .as_slice::<u8>()
                        .context("apply_anchor_restore: map recurrent source")?;
                    let destination = copy
                        .recurrent
                        .destination
                        .as_mut_slice::<u8>()
                        .context("apply_anchor_restore: map recurrent destination")?;
                    destination.copy_from_slice(source);
                }
            }
            LaRestoreRoute::Gpu => {
                let mut encoder = self
                    .device
                    .command_encoder()
                    .context("apply_anchor_restore: create command encoder")?;
                self.encode_la_restore(&mut encoder, &plan.copies)?;
                encoder
                    .commit_and_wait_labeled("qwen35.anchor_restore")
                    .context("apply_anchor_restore: GPU restore")?;
            }
        }
        let slot_index = plan.slot.0 as usize;
        for (layer, saved) in self.full_attn.iter_mut().zip(plan.cursors) {
            layer.current_len[slot_index] = saved;
        }
        if let (Some(mtp), Some(saved)) = (self.mtp_slot.as_mut(), plan.mtp_cursor) {
            mtp.current_len[slot_index] = saved;
        }
        for layer in &mut self.linear_attn {
            layer.pp_flipped[slot_index] = false;
        }
        Ok(())
    }

    /// Validate every layer and destination before any GPU command is encoded.
    pub(crate) fn plan_la_restore(&self, items: &[LaRestoreItem]) -> Result<LaRestorePlan> {
        for (index, item) in items.iter().enumerate() {
            if item.slot.0 >= self.n_seqs {
                return Err(anyhow!(
                    "plan_la_restore: SlotOutOfRange slot={} max_slots={}",
                    item.slot.0,
                    self.n_seqs
                ));
            }
            if items[..index].iter().any(|prior| prior.slot == item.slot) {
                return Err(anyhow!(
                    "plan_la_restore: duplicate destination slot {}",
                    item.slot.0
                ));
            }
        }
        if items.is_empty() {
            return Ok(LaRestorePlan { copies: Vec::new() });
        }

        let copy_count = self
            .linear_attn
            .len()
            .checked_mul(items.len())
            .ok_or_else(|| anyhow!("plan_la_restore: copy-count overflow"))?;
        let mut copies = Vec::with_capacity(copy_count);
        let n_seqs = self.n_seqs as usize;

        for (layer_index, layer) in self.linear_attn.iter().enumerate() {
            let capture = layer.capture_states.as_ref().ok_or_else(|| {
                anyhow!("plan_la_restore: linear_attn[{layer_index}].capture_states is absent")
            })?;
            let conv_capture = layer.conv_capture_states.as_ref().ok_or_else(|| {
                anyhow!("plan_la_restore: linear_attn[{layer_index}].conv_capture_states is absent")
            })?;

            let recurrent_total = layer.recurrent.element_count();
            if recurrent_total == 0 || recurrent_total % n_seqs != 0 {
                return Err(anyhow!(
                    "plan_la_restore: linear_attn[{layer_index}] recurrent elements {recurrent_total} are not a positive multiple of n_seqs {n_seqs}"
                ));
            }
            let recurrent_per_slot = recurrent_total / n_seqs;
            let capture_denominator = recurrent_per_slot
                .checked_mul(n_seqs)
                .ok_or_else(|| anyhow!("plan_la_restore: recurrent denominator overflow"))?;
            if capture.element_count() % capture_denominator != 0 {
                return Err(anyhow!(
                    "plan_la_restore: linear_attn[{layer_index}] capture geometry is not slot/token aligned"
                ));
            }
            let capture_rows = capture.element_count() / capture_denominator;

            let conv_shape = layer.conv_state.shape();
            if conv_shape.len() != 3 || conv_shape[2] != n_seqs {
                return Err(anyhow!(
                    "plan_la_restore: linear_attn[{layer_index}] conv_state shape {:?} is not [channels, K-1, n_seqs={n_seqs}]",
                    conv_shape
                ));
            }
            let channels = conv_shape[0];
            let k_minus_one = conv_shape[1];
            let conv_per_slot = channels
                .checked_mul(k_minus_one)
                .ok_or_else(|| anyhow!("plan_la_restore: conv geometry overflow"))?;
            if conv_per_slot == 0
                || layer.conv_state.element_count()
                    != conv_per_slot
                        .checked_mul(n_seqs)
                        .ok_or_else(|| anyhow!("plan_la_restore: conv slot geometry overflow"))?
            {
                return Err(anyhow!(
                    "plan_la_restore: linear_attn[{layer_index}] conv state geometry is inconsistent"
                ));
            }
            let conv_denominator = conv_per_slot
                .checked_mul(n_seqs)
                .ok_or_else(|| anyhow!("plan_la_restore: conv denominator overflow"))?;
            if conv_capture.element_count() % conv_denominator != 0 {
                return Err(anyhow!(
                    "plan_la_restore: linear_attn[{layer_index}] conv capture geometry is not slot/token aligned"
                ));
            }
            let conv_capture_rows = conv_capture.element_count() / conv_denominator;
            if conv_capture_rows != capture_rows {
                return Err(anyhow!(
                    "plan_la_restore: linear_attn[{layer_index}] recurrent capture depth {capture_rows} differs from conv depth {conv_capture_rows}"
                ));
            }

            for item in items {
                let row = (item.slot.0 as usize)
                    .checked_mul(capture_rows)
                    .and_then(|base| base.checked_add(item.capture_row as usize))
                    .ok_or_else(|| anyhow!("plan_la_restore: capture row overflow"))?;
                if item.capture_row as usize >= capture_rows {
                    return Err(anyhow!(
                        "plan_la_restore: capture row {} >= depth {} for linear_attn[{layer_index}]",
                        item.capture_row,
                        capture_rows
                    ));
                }
                let slot = item.slot.0 as usize;
                let recurrent_source = f32_view(
                    capture,
                    row.checked_mul(recurrent_per_slot)
                        .ok_or_else(|| anyhow!("plan_la_restore: recurrent source overflow"))?,
                    recurrent_per_slot,
                    "plan_la_restore recurrent source",
                )?;
                let recurrent_destination = f32_view(
                    current_recurrent(layer, slot),
                    slot.checked_mul(recurrent_per_slot).ok_or_else(|| {
                        anyhow!("plan_la_restore: recurrent destination overflow")
                    })?,
                    recurrent_per_slot,
                    "plan_la_restore recurrent destination",
                )?;
                if views_overlap(&recurrent_source, &recurrent_destination)? {
                    return Err(anyhow!(
                        "plan_la_restore: linear_attn[{layer_index}] recurrent source overlaps destination"
                    ));
                }
                let recurrent_bytes = recurrent_per_slot
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| anyhow!("plan_la_restore: recurrent byte length overflow"))?;
                let recurrent = RecurrentCopy {
                    source: recurrent_source,
                    destination: recurrent_destination,
                    bytes: u64::try_from(recurrent_bytes)
                        .context("plan_la_restore: recurrent byte length does not fit u64")?,
                };
                let conv_source = f32_view(
                    conv_capture,
                    row.checked_mul(conv_per_slot)
                        .ok_or_else(|| anyhow!("plan_la_restore: conv source overflow"))?,
                    conv_per_slot,
                    "plan_la_restore conv source",
                )?;
                let conv_destination = f32_view(
                    current_conv(layer, slot),
                    slot.checked_mul(conv_per_slot)
                        .ok_or_else(|| anyhow!("plan_la_restore: conv destination overflow"))?,
                    conv_per_slot,
                    "plan_la_restore conv destination",
                )?;
                if views_overlap(&conv_source, &conv_destination)? {
                    return Err(anyhow!(
                        "plan_la_restore: linear_attn[{layer_index}] conv source overlaps destination"
                    ));
                }
                let conv = ConvCopy {
                    source: conv_source,
                    destination: conv_destination,
                    channels: u32::try_from(channels)
                        .context("plan_la_restore: channels do not fit u32")?,
                    k_minus_one: u32::try_from(k_minus_one)
                        .context("plan_la_restore: K-1 does not fit u32")?,
                };
                copies.push(RestoreCopy { conv, recurrent });
            }
        }

        Ok(LaRestorePlan { copies })
    }

    /// Encode compute copies first, then recurrent blits, matching the P4a
    /// ordering that passed both metallib paths.
    pub(crate) fn encode_la_restore(
        &mut self,
        encoder: &mut CommandEncoder,
        plan: &LaRestorePlan,
    ) -> Result<()> {
        let registry = self
            .la_restore_registry
            .get_or_insert_with(KernelRegistry::new);
        for copy in &plan.copies {
            let copy = &copy.conv;
            dispatch_strided_copy_f32(
                encoder,
                registry,
                self.device.metal_device(),
                &copy.source,
                &copy.destination,
                &StridedCopyParams {
                    rows: copy.channels,
                    cols: copy.k_minus_one,
                    stride_row: 1,
                    stride_col: copy.channels,
                },
            )
            .context("encode_la_restore: conv strided copy")?;
        }
        for copy in &plan.copies {
            let copy = &copy.recurrent;
            encoder
                .blit_copy_bytes(&copy.source, 0, &copy.destination, 0, copy.bytes)
                .context("encode_la_restore: recurrent blit")?;
        }
        Ok(())
    }

    /// Restore one speculative capture row through the configured route.
    pub(crate) fn rollback_la(&mut self, slot: SlotId, capture_row: u32) -> Result<()> {
        match self.la_restore_route {
            LaRestoreRoute::Host => self.rollback_la_to(slot, capture_row),
            LaRestoreRoute::Gpu => {
                let plan = self.plan_la_restore(&[LaRestoreItem { slot, capture_row }])?;
                if plan.copies.is_empty() {
                    return Ok(());
                }
                let mut encoder = self
                    .device
                    .command_encoder()
                    .context("rollback_la: create command encoder")?;
                self.encode_la_restore(&mut encoder, &plan)?;
                encoder
                    .commit_and_wait_labeled("qwen35.la_restore")
                    .context("rollback_la: GPU capture restore")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35MoeConfig};

    fn config(variant: Qwen35Variant) -> Qwen35Config {
        Qwen35Config {
            variant,
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            layer_types: default_layer_types(4, 2),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 8,
            mrope_section: [2, 2, 2, 2],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 4096,
            vocab_size: 256,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: (variant == Qwen35Variant::Dense).then_some(128),
            moe: (variant == Qwen35Variant::Moe).then_some(Qwen35MoeConfig {
                moe_intermediate_size: 16,
                num_experts: 4,
                num_experts_per_tok: 2,
                shared_expert_intermediate_size: 16,
            }),
        }
    }

    fn plant(cache: &mut HybridKvCache) {
        const BIT_CLASSES: [u32; 12] = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0001,
            0x7fa0_0001,
            0x3f80_0000,
            0xbf80_0000,
        ];
        for (layer_index, layer) in cache.linear_attn.iter_mut().enumerate() {
            layer.pp_flipped = vec![false, true, false, true];
            for (buffer_index, buffer) in [
                layer.capture_states.as_mut().expect("capture"),
                layer.conv_capture_states.as_mut().expect("conv capture"),
                &mut layer.recurrent,
                &mut layer.recurrent_scratch,
                &mut layer.conv_state,
                &mut layer.conv_state_scratch,
            ]
            .into_iter()
            .enumerate()
            {
                for (index, value) in buffer
                    .as_mut_slice::<u32>()
                    .expect("writable test buffer")
                    .iter_mut()
                    .enumerate()
                {
                    *value = BIT_CLASSES[(index + layer_index + buffer_index) % BIT_CLASSES.len()];
                }
            }
        }
    }

    fn assert_state_equal(left: &HybridKvCache, right: &HybridKvCache) {
        assert_eq!(left.linear_attn.len(), right.linear_attn.len());
        for (left, right) in left.linear_attn.iter().zip(&right.linear_attn) {
            assert_eq!(
                left.recurrent.as_slice::<u32>().unwrap(),
                right.recurrent.as_slice::<u32>().unwrap()
            );
            assert_eq!(
                left.recurrent_scratch.as_slice::<u32>().unwrap(),
                right.recurrent_scratch.as_slice::<u32>().unwrap()
            );
            assert_eq!(
                left.conv_state.as_slice::<u32>().unwrap(),
                right.conv_state.as_slice::<u32>().unwrap()
            );
            assert_eq!(
                left.conv_state_scratch.as_slice::<u32>().unwrap(),
                right.conv_state_scratch.as_slice::<u32>().unwrap()
            );
            assert_eq!(left.pp_flipped, right.pp_flipped);
        }
    }

    #[test]
    fn route_resolution_is_explicit_and_moe_defaults_to_host() {
        assert_eq!(
            LaRestoreRoute::resolve(None, Qwen35Variant::Dense).unwrap(),
            LaRestoreRoute::Gpu
        );
        assert_eq!(
            LaRestoreRoute::resolve(None, Qwen35Variant::Moe).unwrap(),
            LaRestoreRoute::Host
        );
        for variant in [Qwen35Variant::Dense, Qwen35Variant::Moe] {
            assert_eq!(
                LaRestoreRoute::resolve(Some("0"), variant).unwrap(),
                LaRestoreRoute::Host
            );
            assert_eq!(
                LaRestoreRoute::resolve(Some("1"), variant).unwrap(),
                LaRestoreRoute::Gpu
            );
            assert!(LaRestoreRoute::resolve(Some("gpu"), variant).is_err());
            assert!(LaRestoreRoute::resolve(Some(""), variant).is_err());
        }
    }

    fn active_slot_state(cache: &HybridKvCache, slot: usize) -> Vec<[Vec<u32>; 2]> {
        let n_seqs = cache.n_seqs as usize;
        cache
            .linear_attn
            .iter()
            .map(|layer| {
                let recurrent = current_recurrent(layer, slot)
                    .as_slice::<u32>()
                    .expect("mapped recurrent");
                let recurrent_per_slot = recurrent.len() / n_seqs;
                let recurrent_start = slot * recurrent_per_slot;
                let conv = current_conv(layer, slot)
                    .as_slice::<u32>()
                    .expect("mapped conv");
                let conv_per_slot = conv.len() / n_seqs;
                let conv_start = slot * conv_per_slot;
                [
                    recurrent[recurrent_start..recurrent_start + recurrent_per_slot].to_vec(),
                    conv[conv_start..conv_start + conv_per_slot].to_vec(),
                ]
            })
            .collect()
    }

    fn perturb(cache: &mut HybridKvCache, target: usize) {
        for layer in &mut cache.linear_attn {
            for buffer in [
                &mut layer.recurrent,
                &mut layer.recurrent_scratch,
                &mut layer.conv_state,
                &mut layer.conv_state_scratch,
            ] {
                for value in buffer.as_mut_slice::<u32>().expect("mapped state") {
                    *value = value.wrapping_add(0x0101_0101);
                }
            }
            layer.pp_flipped[target] = !layer.pp_flipped[target];
        }
    }

    #[test]
    fn anchor_image_round_trips_every_slot_parity_and_route_without_touching_peers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = mlx_native::MlxDevice::new().expect("Metal device");
        let cfg = config(Qwen35Variant::Dense);
        for snapshot_route in [LaRestoreRoute::Host, LaRestoreRoute::Gpu] {
            for restore_route in [LaRestoreRoute::Host, LaRestoreRoute::Gpu] {
                for flipped in [false, true] {
                    for slot in 0..4 {
                        let mut cache =
                            HybridKvCache::new(&cfg, &device, 64, 4).expect("anchor cache");
                        cache.ensure_la_capture(&cfg, &device, 4).unwrap();
                        plant(&mut cache);
                        for layer in &mut cache.linear_attn {
                            layer.pp_flipped[slot] = flipped;
                        }
                        let expected = active_slot_state(&cache, slot);
                        cache.set_la_restore_route(snapshot_route);
                        let image = cache
                            .snapshot_la_anchor(SlotId(slot as u32))
                            .expect("anchor snapshot");
                        let expected_bytes = cache
                            .linear_attn
                            .iter()
                            .map(|layer| {
                                (layer.recurrent.data_byte_len() + layer.conv_state.data_byte_len())
                                    / cache.n_seqs as usize
                            })
                            .sum::<usize>();
                        assert_eq!(image.total_bytes(), expected_bytes);
                        assert!(image.is_right_sized());

                        perturb(&mut cache, slot);
                        let peers_before: Vec<_> = (0..4)
                            .filter(|&peer| peer != slot)
                            .map(|peer| (peer, active_slot_state(&cache, peer)))
                            .collect();
                        let cursors = vec![0; cache.full_attn.len()];
                        cache.set_la_restore_route(restore_route);
                        let plan = cache
                            .plan_anchor_restore(SlotId(slot as u32), &image, &cursors, None)
                            .expect("anchor restore preflight");
                        cache
                            .apply_anchor_restore(plan)
                            .expect("anchor restore apply");
                        assert_eq!(active_slot_state(&cache, slot), expected);
                        for (peer, before) in peers_before {
                            assert_eq!(
                                active_slot_state(&cache, peer),
                                before,
                                "peer slot {peer} changed"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gpu_restore_matches_host_and_preflight_is_atomic() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = mlx_native::MlxDevice::new().expect("Metal device");
        let cfg = config(Qwen35Variant::Dense);
        let mut host = HybridKvCache::new(&cfg, &device, 64, 4).expect("host cache");
        let mut gpu = HybridKvCache::new(&cfg, &device, 64, 4).expect("gpu cache");
        assert_eq!(host.la_restore_route(), LaRestoreRoute::Host);
        assert_eq!(gpu.la_restore_route(), LaRestoreRoute::Host);
        host.ensure_la_capture(&cfg, &device, 4).unwrap();
        gpu.ensure_la_capture(&cfg, &device, 4).unwrap();
        plant(&mut host);
        plant(&mut gpu);

        host.rollback_la_to(SlotId(1), 2).expect("host restore");
        gpu.set_la_restore_route(LaRestoreRoute::Gpu);
        assert_eq!(gpu.la_restore_route(), LaRestoreRoute::Gpu);
        gpu.rollback_la(SlotId(1), 2).expect("GPU restore");
        assert_state_equal(&host, &gpu);

        let before: Vec<[Vec<u32>; 4]> = gpu
            .linear_attn
            .iter()
            .map(|layer| {
                [
                    layer.recurrent.as_slice::<u32>().unwrap().to_vec(),
                    layer.recurrent_scratch.as_slice::<u32>().unwrap().to_vec(),
                    layer.conv_state.as_slice::<u32>().unwrap().to_vec(),
                    layer.conv_state_scratch.as_slice::<u32>().unwrap().to_vec(),
                ]
            })
            .collect();
        let last = gpu.linear_attn.last_mut().expect("linear layer");
        let original = last.conv_capture_states.take().expect("conv capture");
        last.conv_capture_states = Some(original.slice_view(0, original.element_count() - 1));
        assert!(gpu.rollback_la(SlotId(0), 0).is_err());
        for (layer, before) in gpu.linear_attn.iter().zip(before) {
            assert_eq!(layer.recurrent.as_slice::<u32>().unwrap(), before[0]);
            assert_eq!(
                layer.recurrent_scratch.as_slice::<u32>().unwrap(),
                before[1]
            );
            assert_eq!(layer.conv_state.as_slice::<u32>().unwrap(), before[2]);
            assert_eq!(
                layer.conv_state_scratch.as_slice::<u32>().unwrap(),
                before[3]
            );
        }

        let empty = gpu.plan_la_restore(&[]).expect("full acceptance no-op");
        assert!(empty.copies.is_empty());
    }
    #[test]
    fn gpu_restore_matches_host_for_every_slot_row_and_parity() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = mlx_native::MlxDevice::new().expect("Metal device");
        let cfg = config(Qwen35Variant::Dense);

        for flipped in [false, true] {
            for slot in 0..4 {
                for capture_row in 0..4 {
                    let mut host = HybridKvCache::new(&cfg, &device, 64, 4).expect("host cache");
                    let mut gpu = HybridKvCache::new(&cfg, &device, 64, 4).expect("GPU cache");
                    host.ensure_la_capture(&cfg, &device, 4).unwrap();
                    gpu.ensure_la_capture(&cfg, &device, 4).unwrap();
                    plant(&mut host);
                    plant(&mut gpu);
                    for layer in &mut host.linear_attn {
                        layer.pp_flipped[slot] = flipped;
                    }
                    for layer in &mut gpu.linear_attn {
                        layer.pp_flipped[slot] = flipped;
                    }

                    host.rollback_la_to(SlotId(slot as u32), capture_row)
                        .expect("host restore");
                    gpu.set_la_restore_route(LaRestoreRoute::Gpu);
                    gpu.rollback_la(SlotId(slot as u32), capture_row)
                        .expect("GPU restore");
                    assert_state_equal(&host, &gpu);
                }
            }
        }
    }

    #[test]
    fn multi_item_plan_matches_sequential_host_restores() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = mlx_native::MlxDevice::new().expect("Metal device");
        let cfg = config(Qwen35Variant::Dense);
        let orders = [
            vec![
                LaRestoreItem {
                    slot: SlotId(0),
                    capture_row: 3,
                },
                LaRestoreItem {
                    slot: SlotId(2),
                    capture_row: 1,
                },
            ],
            vec![
                LaRestoreItem {
                    slot: SlotId(3),
                    capture_row: 0,
                },
                LaRestoreItem {
                    slot: SlotId(1),
                    capture_row: 2,
                },
                LaRestoreItem {
                    slot: SlotId(0),
                    capture_row: 3,
                },
            ],
            vec![
                LaRestoreItem {
                    slot: SlotId(3),
                    capture_row: 0,
                },
                LaRestoreItem {
                    slot: SlotId(2),
                    capture_row: 1,
                },
                LaRestoreItem {
                    slot: SlotId(1),
                    capture_row: 2,
                },
                LaRestoreItem {
                    slot: SlotId(0),
                    capture_row: 3,
                },
            ],
        ];

        for items in orders {
            let mut host = HybridKvCache::new(&cfg, &device, 64, 4).expect("host cache");
            let mut gpu = HybridKvCache::new(&cfg, &device, 64, 4).expect("GPU cache");
            host.ensure_la_capture(&cfg, &device, 4).unwrap();
            gpu.ensure_la_capture(&cfg, &device, 4).unwrap();
            plant(&mut host);
            plant(&mut gpu);
            for item in &items {
                host.rollback_la_to(item.slot, item.capture_row)
                    .expect("sequential host restore");
            }

            let plan = gpu.plan_la_restore(&items).expect("multi-item plan");
            let mut encoder = device.command_encoder().expect("command encoder");
            gpu.encode_la_restore(&mut encoder, &plan)
                .expect("encode multi-item restore");
            encoder
                .commit_and_wait_labeled("qwen35.la_restore.test_many")
                .expect("multi-item restore wait");
            assert_state_equal(&host, &gpu);
        }
    }
}
