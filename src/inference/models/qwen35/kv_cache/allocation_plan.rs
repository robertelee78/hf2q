//! Checked production layouts. Requested payload is not physical residency/admission.
use super::*;
#[cfg(test)]
#[path = "allocation_plan_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Init {
    Overwrite,
    Zero,
}
#[derive(Clone, Debug)]
pub(super) struct BufferPlan {
    pub label: &'static str,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub bytes: usize,
    pub init: Init,
}
pub(super) fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b).context("cache shape product overflow")
    })
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).context("cache aggregate overflow")
}
impl BufferPlan {
    fn new(label: &'static str, dtype: DType, shape: Vec<usize>, init: Init) -> Result<Self> {
        let elements = product(&shape)?;
        let bytes = product(&[elements, dtype.size_of()])?;
        Ok(Self {
            label,
            dtype,
            shape,
            elements,
            bytes,
            init,
        })
    }
    pub fn allocate(&self, device: &MlxDevice) -> Result<MlxBuffer> {
        // Overwrite roots remain invisible until the owning cursor advances.
        let result = match self.init {
            Init::Overwrite => unsafe {
                device.alloc_buffer_for_overwrite(self.bytes, self.dtype, self.shape.clone())
            },
            Init::Zero => device.alloc_buffer(self.bytes, self.dtype, self.shape.clone()),
        };
        result.map_err(|e| anyhow!("{}: {e}", self.label))
    }
}
fn reserve<T>(n: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .context("cache plan metadata reservation failed")?;
    Ok(result)
}
/// Traversal is shared by Metal allocation and model-free recorder/fault controls.
pub(super) fn allocate_roots_with<T>(
    roots: &[BufferPlan],
    mut allocate: impl FnMut(&BufferPlan) -> Result<T>,
) -> Result<Vec<T>> {
    let mut values = reserve(roots.len())?;
    for root in roots {
        values.push(allocate(root)?);
    }
    Ok(values)
}
fn channels(cfg: &Qwen35Config) -> Result<u32> {
    let key = cfg
        .linear_num_key_heads
        .checked_mul(cfg.linear_key_head_dim)
        .and_then(|v| v.checked_mul(2))
        .context("cache conv key channels overflow")?;
    let value = cfg
        .linear_num_value_heads
        .checked_mul(cfg.linear_value_head_dim)
        .context("cache conv value channels overflow")?;
    key.checked_add(value)
        .context("cache conv channels overflow")
}
#[derive(Clone, Debug)]
pub(super) struct SlotPlan {
    pub slot: LayerSlot,
    pub mtp: bool,
    pub roots: Vec<BufferPlan>,
    pub context: String,
    pub norms: u32,
}
#[derive(Debug)]
pub(super) struct HybridCacheAllocationPlan {
    pub slots: Vec<SlotPlan>,
    pub conv_channels: u32,
    pub full_bytes: usize,
    pub linear_bytes: usize,
    pub total_bytes: usize,
    pub root_count: usize,
}
pub(super) fn tq_roots(
    cfg: &Qwen35Config,
    capacity: u32,
    nseq: u32,
) -> Result<(Vec<BufferPlan>, u32)> {
    if capacity == 0 {
        return Err(anyhow!(
            "alloc_tq_full_attn_buffers: max_seq_len must be > 0"
        ));
    }
    if nseq == 0 {
        return Err(anyhow!("alloc_tq_full_attn_buffers: n_seqs must be > 0"));
    }
    let shape = vec![
        nseq as usize,
        cfg.num_key_value_heads as usize,
        capacity as usize,
        cfg.head_dim as usize,
    ];
    let norms = tq_norms_per_pos_for(cfg.head_dim);
    let norm_shape = vec![
        nseq as usize,
        cfg.num_key_value_heads as usize,
        capacity as usize,
        norms as usize,
    ];
    Ok((
        vec![
            BufferPlan::new(
                "alloc TQ full-attn K packed",
                DType::U8,
                shape.clone(),
                Init::Overwrite,
            )?,
            BufferPlan::new(
                "alloc TQ full-attn K norms",
                DType::F32,
                norm_shape.clone(),
                Init::Overwrite,
            )?,
            BufferPlan::new(
                "alloc TQ full-attn V packed",
                DType::U8,
                shape,
                Init::Overwrite,
            )?,
            BufferPlan::new(
                "alloc TQ full-attn V norms",
                DType::F32,
                norm_shape,
                Init::Overwrite,
            )?,
        ],
        norms,
    ))
}
impl HybridCacheAllocationPlan {
    pub fn new(
        cfg: &Qwen35Config,
        capacity: u32,
        nseq: u32,
        tq: bool,
        include_mtp: bool,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(anyhow!("HybridKvCache: max_seq_len must be > 0"));
        }
        if nseq == 0 {
            return Err(anyhow!("HybridKvCache: n_seqs must be > 0"));
        }
        let conv_channels = channels(cfg)?;
        let history = cfg.linear_conv_kernel_dim.saturating_sub(1).max(1) as usize;
        let mtp = include_mtp && cfg.mtp_num_hidden_layers > 0;
        let count = add(cfg.layer_types.len(), usize::from(mtp))?;
        let mut slots = reserve(count)?;
        let (mut full, mut linear, mut full_bytes, mut linear_bytes, mut root_count) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for index in 0..count {
            let is_mtp = index == cfg.layer_types.len();
            let kind = if is_mtp {
                Qwen35LayerKind::FullAttention
            } else {
                cfg.layer_types[index]
            };
            let (slot, roots, norms, context) = match kind {
                Qwen35LayerKind::FullAttention => {
                    let rank = u32::try_from(full).context("cache full-attention rank overflow")?;
                    full = add(full, 1)?;
                    let (roots, norms) = if tq {
                        tq_roots(cfg, capacity, nseq)?
                    } else {
                        let shape = vec![
                            nseq as usize,
                            cfg.num_key_value_heads as usize,
                            capacity as usize,
                            cfg.head_dim as usize,
                        ];
                        (
                            vec![
                                BufferPlan::new(
                                    "alloc full-attn K",
                                    DType::F32,
                                    shape.clone(),
                                    Init::Overwrite,
                                )?,
                                BufferPlan::new(
                                    "alloc full-attn V",
                                    DType::F32,
                                    shape,
                                    Init::Overwrite,
                                )?,
                            ],
                            0,
                        )
                    };
                    for root in &roots {
                        full_bytes = add(full_bytes, root.bytes)?;
                    }
                    let context = match (is_mtp, tq) {
                        (true, true) => "alloc tq full-attn buffers (MTP slot)".to_owned(),
                        (true, false) => "alloc MTP full-attn slot".to_owned(),
                        (false, true) => format!("alloc tq full-attn buffers (layer {index})"),
                        (false, false) => format!("alloc full-attn slot (layer {index})"),
                    };
                    (LayerSlot::Full(rank), roots, norms, context)
                }
                Qwen35LayerKind::LinearAttention => {
                    let rank =
                        u32::try_from(linear).context("cache linear-attention rank overflow")?;
                    linear = add(linear, 1)?;
                    let conv = vec![conv_channels as usize, history, nseq as usize];
                    let rec = vec![
                        cfg.linear_key_head_dim as usize,
                        cfg.linear_value_head_dim as usize,
                        cfg.linear_num_value_heads as usize,
                        nseq as usize,
                    ];
                    let roots = vec![
                        BufferPlan::new("alloc conv_state", DType::F32, conv.clone(), Init::Zero)?,
                        BufferPlan::new("alloc conv_state_scratch", DType::F32, conv, Init::Zero)?,
                        BufferPlan::new("alloc recurrent", DType::F32, rec.clone(), Init::Zero)?,
                        BufferPlan::new("alloc recurrent_scratch", DType::F32, rec, Init::Zero)?,
                    ];
                    for root in &roots {
                        linear_bytes = add(linear_bytes, root.bytes)?;
                    }
                    (
                        LayerSlot::Linear(rank),
                        roots,
                        0,
                        format!("alloc linear-attn slot (layer {index})"),
                    )
                }
            };
            root_count = add(root_count, roots.len())?;
            slots.push(SlotPlan {
                slot,
                mtp: is_mtp,
                roots,
                context,
                norms,
            });
        }
        Ok(Self {
            slots,
            conv_channels,
            full_bytes,
            linear_bytes,
            total_bytes: add(full_bytes, linear_bytes)?,
            root_count,
        })
    }
    pub fn allocate_with<T>(
        &self,
        mut allocate: impl FnMut(&BufferPlan) -> Result<T>,
    ) -> Result<Vec<Vec<T>>> {
        debug_assert_eq!(self.total_bytes, self.full_bytes + self.linear_bytes);
        debug_assert_eq!(
            self.root_count,
            self.slots.iter().map(|s| s.roots.len()).sum::<usize>()
        );
        let mut output = reserve(self.slots.len())?;
        // Reserve all result-container metadata before the first Metal root.
        for slot in &self.slots {
            output.push(reserve(slot.roots.len())?);
        }
        for (slot, values) in self.slots.iter().zip(&mut output) {
            for root in &slot.roots {
                values.push(allocate(root).with_context(|| slot.context.clone())?);
            }
        }
        Ok(output)
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct ExistingCapture {
    pub elements: usize,
    pub backing_bytes: usize,
}
impl ExistingCapture {
    pub fn from_buffer(buffer: &MlxBuffer) -> Result<Self> {
        Ok(Self {
            elements: product(buffer.shape())?,
            backing_bytes: buffer.byte_len(),
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CaptureRole {
    Recurrent,
    Conv,
}
#[derive(Debug)]
pub(super) struct CaptureAction {
    pub slot: usize,
    pub role: CaptureRole,
    pub allocation: Option<BufferPlan>,
}
#[derive(Debug)]
pub(super) struct CaptureAllocationPlan {
    pub actions: Vec<CaptureAction>,
    pub old_backing_bytes: usize,
    pub new_bytes: usize,
    pub retained_after_bytes: usize,
    pub old_plus_new_bytes: usize,
    pub old_refs: usize,
    pub new_roots: usize,
    pub upper_refs: usize,
}
impl CaptureAllocationPlan {
    pub fn new(
        cfg: &Qwen35Config,
        nseq: u32,
        tokens: u32,
        existing: &[[Option<ExistingCapture>; 2]],
    ) -> Result<Self> {
        if tokens == 0 {
            return Err(anyhow!("ensure_la_capture: n_tokens_max must be > 0"));
        }
        let channels = channels(cfg)? as usize;
        let rec = BufferPlan::new(
            "alloc capture_states",
            DType::F32,
            vec![
                cfg.linear_key_head_dim as usize,
                cfg.linear_value_head_dim as usize,
                cfg.linear_num_value_heads as usize,
                tokens as usize,
                nseq as usize,
            ],
            Init::Zero,
        )?;
        // Deliberately NOT max(1): zero-byte conv fails in the actual allocator
        // after any successful recurrent assignment, preserving existing behavior.
        let conv = BufferPlan::new(
            "alloc conv_capture_states",
            DType::F32,
            vec![
                nseq as usize,
                tokens as usize,
                cfg.linear_conv_kernel_dim.saturating_sub(1) as usize,
                channels,
            ],
            Init::Zero,
        )?;
        let mut actions = reserve(product(&[existing.len(), 2])?)?;
        let (
            mut old_backing_bytes,
            mut new_bytes,
            mut retained_after_bytes,
            mut old_refs,
            mut new_roots,
        ) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for (slot, old) in existing.iter().enumerate() {
            for (index, descriptor) in [&rec, &conv].into_iter().enumerate() {
                let keep = old[index].is_some_and(|o| {
                    (nseq == 1 && o.elements >= descriptor.elements)
                        || o.elements == descriptor.elements
                });
                if let Some(o) = old[index] {
                    old_backing_bytes = add(old_backing_bytes, o.backing_bytes)?;
                    old_refs = add(old_refs, 1)?;
                }
                let allocation = if keep {
                    retained_after_bytes =
                        add(retained_after_bytes, old[index].unwrap().backing_bytes)?;
                    None
                } else {
                    new_bytes = add(new_bytes, descriptor.bytes)?;
                    new_roots = add(new_roots, 1)?;
                    retained_after_bytes = add(retained_after_bytes, descriptor.bytes)?;
                    Some(descriptor.clone())
                };
                actions.push(CaptureAction {
                    slot,
                    role: if index == 0 {
                        CaptureRole::Recurrent
                    } else {
                        CaptureRole::Conv
                    },
                    allocation,
                });
            }
        }
        Ok(Self {
            actions,
            old_backing_bytes,
            new_bytes,
            retained_after_bytes,
            old_plus_new_bytes: add(old_backing_bytes, new_bytes)?,
            old_refs,
            new_roots,
            upper_refs: add(old_refs, new_roots)?,
        })
    }
    pub fn apply_with<T>(
        &self,
        mut allocate: impl FnMut(&BufferPlan) -> Result<T>,
        mut install: impl FnMut(usize, CaptureRole, T),
    ) -> Result<()> {
        debug_assert_eq!(
            self.old_plus_new_bytes,
            self.old_backing_bytes + self.new_bytes
        );
        debug_assert_eq!(self.upper_refs, self.old_refs + self.new_roots);
        debug_assert!(self.retained_after_bytes <= self.old_plus_new_bytes);
        for action in &self.actions {
            if let Some(root) = &action.allocation {
                let value = allocate(root)?;
                install(action.slot, action.role, value);
            }
        }
        Ok(())
    }
}
/// Actual activity state machine, shared with pure preparation/allocation faults.
pub(super) fn ensure_capture_with(
    active: &mut Option<u32>,
    tokens: u32,
    operation: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if tokens == 0 {
        return Err(anyhow!("ensure_la_capture: n_tokens_max must be > 0"));
    }
    *active = None;
    operation()?;
    *active = Some(tokens);
    Ok(())
}
