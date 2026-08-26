//! Activation-time routing for artifact-native scalar expert stacks.
//!
//! Model loaders lend exact F32/F16/BF16 expert buffers while the primary and
//! optional parallel-encode registries are frozen. Plans retain no model
//! buffers; installation into the worker registry must revalidate the same
//! activation cases so route authority cannot cross a model swap.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use mlx_native::{
    calibrate_dense_matmul_id_routes, DType, DenseMatmulIdCalibrationBatchReceipt,
    DenseMatmulIdCalibrationCase, DenseMatmulIdCalibrationLimits, DenseMatmulIdInputLayout,
    DenseMatmulIdMultiplicity, DenseMatmulIdParams, DenseMatmulIdRoute, DenseMatmulIdRoutePlan,
    DenseMatmulIdScratch, DenseMatmulIdShape, KernelRegistry, MlxBuffer, MlxDevice,
};

struct DenseExpertScratchEntry {
    activation_epoch: u64,
    device_registry_id: u64,
    scratch: DenseMatmulIdScratch,
}

/// Thread-local scalar-expert scratch whose ownership is bound to one model
/// activation as well as one physical device. Capacity alone is insufficient:
/// a model swap on the same serving thread must not reuse the prior model's
/// residency-set resources.
#[derive(Default)]
pub(crate) struct DenseExpertScratchCache {
    entry: Option<DenseExpertScratchEntry>,
    #[cfg(test)]
    allocation_generation: u64,
}

impl DenseExpertScratchCache {
    pub(crate) fn owned_bytes(&self) -> u64 {
        self.entry.as_ref().map_or(0, |entry| {
            u64::from(entry.scratch.max_experts())
                .saturating_mul(u64::from(entry.scratch.max_tokens()).saturating_add(1))
                .saturating_mul(std::mem::size_of::<u32>() as u64)
        })
    }

    /// Drop the exact registered scratch owner and return its logical buffer
    /// bytes. Callers may use this only at a drained command-graph boundary.
    pub(crate) fn release_owned_bytes(&mut self) -> u64 {
        let bytes = self.owned_bytes();
        self.entry.take();
        bytes
    }

    #[cfg(test)]
    pub(crate) fn has_entry_for_test(&self) -> bool {
        self.entry.is_some()
    }

    pub(crate) fn with<R>(
        &mut self,
        activation_epoch: u64,
        device: &MlxDevice,
        n_experts: u32,
        max_tokens: u32,
        f: impl FnOnce(&DenseMatmulIdScratch) -> mlx_native::Result<R>,
    ) -> mlx_native::Result<R> {
        if activation_epoch == 0 {
            return Err(mlx_native::MlxError::InvalidArgument(
                "scalar expert scratch activation epoch must be nonzero".into(),
            ));
        }
        let device_registry_id = device.registry_id();
        let must_allocate = self.entry.as_ref().is_none_or(|entry| {
            entry.activation_epoch != activation_epoch
                || entry.device_registry_id != device_registry_id
                || n_experts > entry.scratch.max_experts()
                || max_tokens > entry.scratch.max_tokens()
        });
        if must_allocate {
            // Capacity is not activation authority. Release the prior
            // device/epoch owner before allocating its replacement so a
            // failed A→B rebind cannot keep stale scratch alive or transiently
            // co-reside both activation owners.
            self.entry.take();
            self.entry = Some(DenseExpertScratchEntry {
                activation_epoch,
                device_registry_id,
                scratch: DenseMatmulIdScratch::new(device, n_experts, max_tokens)?,
            });
            #[cfg(test)]
            {
                self.allocation_generation += 1;
            }
        }
        f(&self
            .entry
            .as_ref()
            .expect("scalar expert scratch cache allocated")
            .scratch)
    }
}

#[derive(Clone)]
pub(crate) struct NativeScalarExpertMatrix<'a> {
    pub label: &'a str,
    pub weight: &'a MlxBuffer,
    pub n: u32,
    pub k: u32,
    pub top_k: u32,
    pub n_experts: u32,
    pub expert_stride_bytes: u64,
    pub input_layout: DenseMatmulIdInputLayout,
    pub id_multiplicity: DenseMatmulIdMultiplicity,
    /// Exact physical source-row widths worth timing at activation. Other
    /// widths with the same base contract remain correct through Direct.
    pub calibrated_m: Vec<u32>,
}

impl NativeScalarExpertMatrix<'_> {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.weight.dtype(), DType::F32 | DType::F16 | DType::BF16),
            "{}: scalar expert activation requires native F32/F16/BF16 storage, got {}",
            self.label,
            self.weight.dtype()
        );
        ensure!(
            self.n > 0
                && self.k > 0
                && self.top_k > 0
                && self.n_experts > 0
                && self.expert_stride_bytes > 0,
            "{}: scalar expert dimensions, top-k, count, and stride must be nonzero",
            self.label
        );
        ensure!(
            !self.calibrated_m.is_empty() && self.calibrated_m.iter().all(|m| *m > 0),
            "{}: scalar expert activation must declare nonzero physical row widths",
            self.label
        );
        Ok(())
    }

    fn params(&self, m: u32) -> DenseMatmulIdParams {
        DenseMatmulIdParams {
            m,
            n: self.n,
            k: self.k,
            top_k: self.top_k,
            n_experts: self.n_experts,
            expert_stride_bytes: self.expert_stride_bytes,
            input_layout: self.input_layout,
            id_multiplicity: self.id_multiplicity,
            route: DenseMatmulIdRoute::Direct,
        }
    }
}

pub(crate) struct DenseExpertActivation {
    pub plan: Arc<DenseMatmulIdRoutePlan>,
    pub receipt: DenseMatmulIdCalibrationBatchReceipt,
}

fn activation_cases<'a>(
    matrices: &'a [NativeScalarExpertMatrix<'a>],
) -> Result<(Vec<DenseMatmulIdParams>, Vec<&'a MlxBuffer>)> {
    let mut params = Vec::new();
    let mut weights = Vec::new();
    for matrix in matrices {
        matrix.validate()?;
        let mut widths = matrix.calibrated_m.clone();
        widths.sort_unstable();
        widths.dedup();
        for m in widths {
            params.push(matrix.params(m));
            weights.push(matrix.weight);
        }
    }
    Ok((params, weights))
}

fn borrow_cases<'a>(
    params: &'a [DenseMatmulIdParams],
    weights: &'a [&'a MlxBuffer],
) -> Vec<DenseMatmulIdCalibrationCase<'a>> {
    params
        .iter()
        .zip(weights)
        .map(|(params, weight)| DenseMatmulIdCalibrationCase {
            weight,
            params: *params,
        })
        .collect()
}

/// Calibrate the exact union of scalar expert shapes once and freeze it into
/// the primary registry. The optional worker receives the identical plan only
/// after mlx-native revalidates the same current activation cases.
pub(crate) fn activate_native_scalar_experts(
    registry: &mut KernelRegistry,
    worker_registry: Option<&mut KernelRegistry>,
    device: &MlxDevice,
    activation_epoch: u64,
    matrices: &[NativeScalarExpertMatrix<'_>],
) -> Result<Option<DenseExpertActivation>> {
    ensure!(
        activation_epoch != 0,
        "scalar expert activation epoch must be nonzero"
    );
    if matrices.is_empty() {
        return Ok(None);
    }

    let (params, weights) = activation_cases(matrices)?;
    let cases = borrow_cases(&params, &weights);
    let unique_shapes = cases
        .iter()
        .map(|case| DenseMatmulIdShape {
            weight_dtype: case.weight.dtype(),
            m: case.params.m,
            n: case.params.n,
            k: case.params.k,
            top_k: case.params.top_k,
            n_experts: case.params.n_experts,
            expert_stride_bytes: case.params.expert_stride_bytes,
            input_layout: case.params.input_layout,
            id_multiplicity: case.params.id_multiplicity,
        })
        .collect::<BTreeSet<_>>();
    let declared_cases = u32::try_from(unique_shapes.len())
        .context("scalar expert declared-shape count exceeds calibration API")?;
    let limits = DenseMatmulIdCalibrationLimits::default();
    ensure!(
        declared_cases <= limits.max_cases,
        "scalar expert activation declares {declared_cases} shapes, safety cap is {}",
        limits.max_cases
    );
    let (plan, receipt) =
        calibrate_dense_matmul_id_routes(registry, device, activation_epoch, limits, &cases)
            .context("calibrate native scalar expert routes")?;
    ensure!(
        receipt.declared_cases == declared_cases,
        "scalar expert receipt covered {} shapes, expected {declared_cases}",
        receipt.declared_cases
    );
    ensure!(
        receipt.activation_epoch == activation_epoch
            && plan.activation_epoch() == activation_epoch
            && receipt.plan_id == plan.plan_id()
            && receipt.activation_authority_digest == plan.activation_authority_digest()
            && receipt.declared_cases as usize == plan.decision_count(),
        "scalar expert calibration receipt and frozen plan authority differ"
    );

    if let Some(worker) = worker_registry {
        worker
            .freeze_dense_matmul_id_plan_for_cases(device, activation_epoch, plan.clone(), &cases)
            .context("freeze scalar expert plan into worker registry")?;
    }

    tracing::info!(
        activation_epoch,
        plan_id = plan.plan_id(),
        declared_cases = receipt.declared_cases,
        declared_weight_identities = receipt.declared_weight_identities,
        calibrated_decisions = receipt.calibrated_decisions,
        process_cache_hits = receipt.process_cache_hits,
        fallback_decisions = receipt.fallback_decisions,
        calibration_submissions = receipt.calibration_submissions,
        elapsed_ms = receipt.elapsed_ms,
        "activated native scalar expert routes"
    );
    Ok(Some(DenseExpertActivation { plan, receipt }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_cache_replaces_equal_capacity_resources_across_a_b_a_epochs() {
        let device_a = MlxDevice::new().expect("activation-A Metal device");
        let device_b = MlxDevice::new().expect("activation-B Metal device");
        let mut cache = DenseExpertScratchCache::default();
        for (epoch, device, expected_generation) in [
            (11, &device_a, 1),
            (11, &device_a, 1),
            (22, &device_b, 2),
            (33, &device_a, 3),
        ] {
            cache
                .with(epoch, device, 8, 4, |_| Ok(()))
                .expect("allocate epoch-bound scalar scratch");
            assert_eq!(cache.allocation_generation, expected_generation);
        }
    }
}
