//! Checked requested payload consumed by actual Gemma SlotAware allocators.
//! Excludes loader caches, retired owners, physical residency and admission.
use crate::serve::config::Gemma4Config;
use anyhow::{ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

#[cfg(test)]
#[path = "cache_scaffold_plan_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    Hb,
    Hybrid,
    Dense,
    Nibble,
}
#[derive(Clone, Copy)]
pub(crate) struct Modes {
    pub hybrid: bool,
    pub dense: bool,
    pub nibble: bool,
    pub dense_dtype: DType,
    pub full_f16_v: bool,
    pub xlen: bool,
}
impl Modes {
    pub(crate) fn plain(dense_dtype: DType) -> Self {
        Self {
            hybrid: false,
            dense: false,
            nibble: false,
            dense_dtype,
            full_f16_v: false,
            xlen: false,
        }
    }
    pub(crate) fn hybrid_invocation() -> Self {
        Self {
            full_f16_v: std::env::var("HF2Q_FULL_F16_KV")
                .ok()
                .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "on")),
            xlen: std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1"),
            ..Self::plain(DType::F32)
        }
    }

    // One invocation snapshot; process environment must remain stable.
    pub(crate) fn invocation() -> Self {
        let env = &crate::debug::INVESTIGATION_ENV;
        Self {
            hybrid: env.hybrid_kv,
            dense: env.use_dense,
            nibble: crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits()
                == 0,
            dense_dtype: if env.f16_kv { DType::F16 } else { DType::F32 },
            full_f16_v: std::env::var("HF2Q_FULL_F16_KV")
                .ok()
                .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "on")),
            xlen: std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1"),
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct BufferDescriptor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub zeroed: bool,
    pub error: String,
}
impl BufferDescriptor {
    fn new(dtype: DType, shape: Vec<usize>, zeroed: bool, error: String) -> Result<Self> {
        let bytes = shape
            .iter()
            .try_fold(dtype.size_of(), |n, d| n.checked_mul(*d))
            .context("Gemma scaffold shape/byte overflow")?;
        ensure!(bytes > 0, "Gemma scaffold zero payload");
        Ok(Self {
            dtype,
            shape,
            bytes,
            zeroed,
            error,
        })
    }
    pub(crate) fn allocate(&self, dev: &MlxDevice) -> Result<MlxBuffer> {
        let result = if self.zeroed {
            dev.alloc_buffer(self.bytes, self.dtype, self.shape.clone())
        } else {
            // SAFETY: full-layer readers are bounded by per-slot cursors.
            // Producers write each visible position before advancing its cursor;
            // reset lowers the cursor and never exposes the unwritten tail.
            unsafe { dev.alloc_buffer_for_overwrite(self.bytes, self.dtype, self.shape.clone()) }
        };
        result.map_err(|e| anyhow::anyhow!("{}: {e}", self.error))
    }
}
#[derive(Clone)]
pub(crate) struct SlotPlan {
    pub phase: Phase,
    pub layer: usize,
    pub nkv: usize,
    pub hd: usize,
    pub cap: usize,
    pub ring: bool,
    pub nseq: u32,
    pub norms: usize,
    pub dense_dtype: DType,
    pub xlen: bool,
    pub buffers: Vec<BufferDescriptor>,
    pub bytes: usize,
}
impl SlotPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        phase: Phase,
        layer: usize,
        nkv: usize,
        hd: usize,
        cap: usize,
        ring: bool,
        nseq: u32,
        modes: Modes,
        nibble_norms: Option<usize>,
    ) -> Result<Self> {
        let prefix = match phase {
            Phase::Hb => "alloc_hb_kv_for_layer",
            Phase::Hybrid => "alloc_multi_seq_hybrid_kv_for_layer",
            Phase::Dense => "alloc_multi_seq_dense_kv_for_layer",
            Phase::Nibble => "alloc_multi_seq_mlx_kv_for_layer",
        };
        ensure!(nseq > 0, "{prefix} L{layer}: n_seqs must be > 0");
        let norms = nibble_norms.unwrap_or_else(|| (hd / 256).max(1));
        if phase == Phase::Nibble {
            ensure!(nkv > 0 && hd > 0 && cap > 0 && norms > 0,
                "{prefix} L{layer}: nkv/hd/cap/norms_per_pos must be > 0 (got nkv={nkv}, hd={hd}, cap={cap}, norms_per_pos={norms})");
            ensure!(hd % 2 == 0, "{prefix} L{layer}: hd must be even for 4-bit nibble-packed K/V (hd/2 stride; got hd={hd})");
        } else {
            ensure!(
                nkv > 0 && hd > 0 && cap > 0,
                "{prefix} L{layer}: nkv/hd/cap must be > 0 (got nkv={nkv}, hd={hd}, cap={cap})"
            );
        }
        let n = nseq as usize;
        let cursor = vec![n, nkv, cap, hd];
        let norm_shape = if phase == Phase::Nibble && norms == 1 {
            vec![n, nkv, cap]
        } else {
            vec![n, nkv, cap, norms]
        };
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(5)
            .context("Gemma scaffold descriptor reservation")?;
        let mut add = |dtype, shape, zeroed, label: String| -> Result<()> {
            buffers.push(BufferDescriptor::new(dtype, shape, zeroed, label)?);
            Ok(())
        };
        match phase {
            Phase::Hb | Phase::Nibble => {
                let packed = vec![n, nkv, cap, if phase == Phase::Hb { hd } else { hd / 2 }];
                let label = if phase == Phase::Hb {
                    "hb_kv"
                } else {
                    "multi-seq MLX"
                };
                for (dtype, shape, role) in [
                    (DType::U8, packed.clone(), "K packed"),
                    (DType::F32, norm_shape.clone(), "K norms"),
                    (DType::U8, packed, "V packed"),
                    (DType::F32, norm_shape, "V norms"),
                ] {
                    add(
                        dtype,
                        shape,
                        ring,
                        if phase == Phase::Hb {
                            format!("{label} L{layer} {role}")
                        } else {
                            format!("{label} {role} L{layer}")
                        },
                    )?;
                }
            }
            Phase::Dense => {
                for role in ["K", "V"] {
                    add(
                        modes.dense_dtype,
                        cursor.clone(),
                        ring,
                        format!("multi-seq dense {role} L{layer}"),
                    )?;
                }
            }
            Phase::Hybrid => {
                add(
                    DType::F16,
                    cursor.clone(),
                    ring,
                    format!("multi-seq hybrid F16 K L{layer}"),
                )?;
                if modes.full_f16_v {
                    add(
                        DType::F16,
                        cursor.clone(),
                        ring,
                        format!("multi-seq hybrid F16 V L{layer}"),
                    )?;
                    add(
                        DType::F32,
                        vec![1],
                        true,
                        format!("multi-seq hybrid V norms (dummy) L{layer}"),
                    )?;
                } else {
                    add(
                        DType::U8,
                        cursor.clone(),
                        ring,
                        format!("multi-seq hybrid V packed L{layer}"),
                    )?;
                    add(
                        DType::F32,
                        norm_shape,
                        ring,
                        format!("multi-seq hybrid V norms L{layer}"),
                    )?;
                }
                if modes.xlen {
                    for role in ["K", "V"] {
                        add(
                            DType::BF16,
                            cursor.clone(),
                            ring,
                            format!("multi-seq hybrid bf16 xlen {role} L{layer}"),
                        )?;
                    }
                }
            }
        }
        let bytes = buffers
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.bytes))
            .context("Gemma scaffold slot byte sum overflow")?;
        Ok(Self {
            phase,
            layer,
            nkv,
            hd,
            cap,
            ring,
            nseq,
            norms,
            dense_dtype: modes.dense_dtype,
            xlen: phase == Phase::Hybrid && modes.xlen,
            buffers,
            bytes,
        })
    }
    // Shared ACTUAL descriptor traversal: recorder/fault controls use this too.
    pub(crate) fn allocate_with<T>(
        &self,
        mut allocate: impl FnMut(&BufferDescriptor) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.buffers.len())
            .context("Gemma scaffold allocation record reservation")?;
        for buffer in &self.buffers {
            output.push(allocate(buffer)?);
        }
        Ok(output)
    }
    pub(crate) fn outer_context(&self) -> String {
        let Self {
            layer: i,
            nseq: n,
            nkv,
            hd,
            cap,
            ring,
            ..
        } = self;
        match self.phase {
            Phase::Hb => format!("ADR-040 C2c: alloc_hb_kv_for_layer L{i} failed for max_slots={n} (nkv={nkv}, hd={hd}, cap={cap}, is_ring={ring})"),
            Phase::Hybrid => format!("ADR-040 iter-C2c-cont: alloc_multi_seq_hybrid_kv_for_layer L{i} failed for max_slots={n} (nkv={nkv}, hd={hd}, cap={cap}, is_ring={ring}) — production-default hybrid F16-K + TQ-HB-V regime per H10 falsification §6.1.11"),
            Phase::Dense => format!("ADR-040 iter-C2c-cont-cont: alloc_multi_seq_dense_kv_for_layer L{i} failed for max_slots={n} (nkv={nkv}, hd={hd}, cap={cap}, is_ring={ring}, dtype={:?}) — HF2Q_USE_DENSE=1 opt-in pre-default surface per §6.1.46", self.dense_dtype),
            Phase::Nibble => format!("ADR-040 iter-C2c-cont-cont: alloc_multi_seq_mlx_kv_for_layer L{i} failed for max_slots={n} (nkv={nkv}, hd={hd}, cap={cap}, is_ring={ring}, norms_per_pos={}) — HF2Q_TQ_CODEBOOK_BITS=4 opt-in pre-default surface per §6.1.46",self.norms),
        }
    }
}
pub(crate) struct PhasePlan {
    pub phase: Phase,
    pub slots: Vec<SlotPlan>,
}
pub(crate) struct ScaffoldPlan {
    pub phases: Vec<PhasePlan>,
    pub bytes: usize,
    pub roots: usize,
}
impl ScaffoldPlan {
    pub(crate) fn new(cfg: &Gemma4Config, layers: usize, nseq: u32, modes: Modes) -> Result<Self> {
        ensure!(nseq > 0, "ADR-040 C2c: provision_multi_seq_kv_for_slot_aware called with max_slots == 0; spawn_with_mode invariant is max_slots >= 1 (EngineMode::SlotAware variant enforces this at the API boundary — caller violated)");
        let mut phases = Vec::new();
        let mut bytes = 0usize;
        let mut roots = 0usize;
        for (phase, enabled) in [
            (Phase::Hb, true),
            (Phase::Hybrid, modes.hybrid),
            (Phase::Dense, modes.dense),
            (Phase::Nibble, modes.nibble),
        ] {
            if !enabled {
                continue;
            }
            let mut slots = Vec::new();
            slots
                .try_reserve_exact(layers)
                .context("Gemma scaffold layer record reservation")?;
            for layer in 0..layers {
                let full = cfg.is_full_attention(layer);
                let (ring, cap) = super::layer_type_to_alloc_params_per_slot(
                    if full {
                        crate::serve::config::LayerType::Full
                    } else {
                        crate::serve::config::LayerType::Sliding
                    },
                    cfg.sliding_window,
                    cfg.max_position_embeddings,
                    nseq as usize,
                );
                let slot = SlotPlan::new(
                    phase,
                    layer,
                    cfg.num_kv_heads_for_layer(layer),
                    cfg.head_dim_for_layer(layer),
                    cap,
                    ring,
                    nseq,
                    modes,
                    None,
                )?;
                bytes = bytes
                    .checked_add(slot.bytes)
                    .context("Gemma scaffold aggregate bytes overflow")?;
                roots = roots
                    .checked_add(slot.buffers.len())
                    .context("Gemma scaffold aggregate roots overflow")?;
                slots.push(slot);
            }
            phases.push(PhasePlan { phase, slots });
        }
        Ok(Self {
            phases,
            bytes,
            roots,
        })
    }
    // Same production phase-publication traversal used by pure fault controls.
    pub(crate) fn execute<T>(
        &self,
        mut allocate: impl FnMut(&SlotPlan) -> Result<T>,
        mut publish: impl FnMut(Phase, Vec<T>) -> Result<()>,
    ) -> Result<()> {
        // Reserve all phase result containers before any device allocation.
        let mut containers = Vec::new();
        for phase in &self.phases {
            let mut values = Vec::new();
            values
                .try_reserve_exact(phase.slots.len())
                .context("Gemma scaffold phase result reservation")?;
            containers.push(values);
        }
        for (phase, mut values) in self.phases.iter().zip(containers) {
            for slot in &phase.slots {
                values.push(allocate(slot).with_context(|| slot.outer_context())?);
            }
            publish(phase.phase, values)?;
        }
        Ok(())
    }
}
