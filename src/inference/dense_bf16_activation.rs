//! Shared activation-time routing for artifact-native BF16 dense matrices.
//!
//! Inference call sites consume one immutable registry-local plan. Model
//! loaders provide borrowed native buffers during activation; neither the
//! plan nor the process cache retains model weights.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::{
    calibrate_dense_bf16_routes, DType, DenseBf16BaseShape, DenseBf16CalibrationBatchReceipt,
    DenseBf16CalibrationCase, DenseBf16CalibrationLimits, DenseBf16RoutePlan, KernelRegistry,
    MlxBuffer, MlxDevice,
};

const ALL_SHORT_ROW_MASK: u16 = u16::MAX;

#[derive(Clone, Copy)]
pub(crate) struct NativeBf16Matrix<'a> {
    pub label: &'a str,
    pub weight: &'a MlxBuffer,
    pub n: u32,
    pub k: u32,
    pub src0_batch: u32,
    pub src1_batch: u32,
    /// Bit `m - 1` declares physical row width `m`, for `1 <= m <= 16`.
    pub reachable_row_mask: u16,
}

impl<'a> NativeBf16Matrix<'a> {
    pub(crate) fn unbatched(label: &'a str, weight: &'a MlxBuffer, n: u32, k: u32) -> Self {
        Self {
            label,
            weight,
            n,
            k,
            src0_batch: 1,
            src1_batch: 1,
            reachable_row_mask: ALL_SHORT_ROW_MASK,
        }
    }

    pub(crate) fn unbatched_single_row(
        label: &'a str,
        weight: &'a MlxBuffer,
        n: u32,
        k: u32,
    ) -> Self {
        Self {
            reachable_row_mask: 1,
            ..Self::unbatched(label, weight, n, k)
        }
    }

    pub(crate) fn unbatched_through(
        label: &'a str,
        weight: &'a MlxBuffer,
        n: u32,
        k: u32,
        max_m: u32,
    ) -> Result<Self> {
        ensure!(
            (1..=16).contains(&max_m),
            "{label}: reachable dense width must be in 1..=16, got {max_m}"
        );
        let mask = if max_m == 16 {
            ALL_SHORT_ROW_MASK
        } else {
            (1u16 << max_m) - 1
        };
        Ok(Self {
            reachable_row_mask: mask,
            ..Self::unbatched(label, weight, n, k)
        })
    }

    fn base_shape(self) -> DenseBf16BaseShape {
        DenseBf16BaseShape {
            n: self.n,
            k: self.k,
            src0_batch: self.src0_batch,
            src1_batch: self.src1_batch,
        }
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.weight.dtype() == DType::BF16,
            "{}: dense activation requires native BF16 storage, got {}",
            self.label,
            self.weight.dtype()
        );
        ensure!(
            self.n > 0 && self.k > 0 && self.src0_batch > 0 && self.src1_batch > 0,
            "{}: dense activation dimensions and batch counts must be nonzero",
            self.label
        );
        ensure!(
            self.reachable_row_mask != 0,
            "{}: dense activation declares no reachable row widths",
            self.label
        );
        let elements = usize::try_from(self.n)
            .ok()
            .and_then(|n| usize::try_from(self.k).ok().and_then(|k| n.checked_mul(k)))
            .and_then(|matrix| {
                usize::try_from(self.src0_batch)
                    .ok()
                    .and_then(|batch| matrix.checked_mul(batch))
            })
            .ok_or_else(|| anyhow!("{}: dense activation element count overflows", self.label))?;
        ensure!(
            self.weight.element_count() >= elements,
            "{}: BF16 buffer has {} elements, needs at least {} for [{}, {}] x batch {}",
            self.label,
            self.weight.element_count(),
            elements,
            self.n,
            self.k,
            self.src0_batch
        );
        ensure!(
            self.src1_batch == 1 || self.src1_batch == self.src0_batch,
            "{}: unsupported dense activation batch ratio {}:{}",
            self.label,
            self.src0_batch,
            self.src1_batch
        );
        Ok(())
    }
}

pub(crate) struct DenseBf16Activation {
    pub plan: Arc<DenseBf16RoutePlan>,
    pub receipt: DenseBf16CalibrationBatchReceipt,
}

/// Calibrate every distinct base shape over the physical short-row widths and
/// freeze the resulting plan into `registry` before the model becomes live.
pub(crate) fn activate_native_bf16_dense(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    matrices: &[NativeBf16Matrix<'_>],
) -> Result<Option<DenseBf16Activation>> {
    ensure!(
        activation_epoch != 0,
        "dense activation epoch must be nonzero"
    );

    let mut representatives = BTreeMap::<DenseBf16BaseShape, (&MlxBuffer, u16)>::new();
    for matrix in matrices.iter().copied() {
        matrix.validate()?;
        representatives
            .entry(matrix.base_shape())
            .and_modify(|(_, mask)| *mask |= matrix.reachable_row_mask)
            .or_insert((matrix.weight, matrix.reachable_row_mask));
    }
    if representatives.is_empty() {
        return Ok(None);
    }

    let owned_cases: Vec<_> = representatives
        .into_iter()
        .map(|(shape, (weight, mask))| {
            let reachable_m: Vec<u32> =
                (1..=16).filter(|m| mask & (1u16 << (m - 1)) != 0).collect();
            (shape, weight, reachable_m)
        })
        .collect();
    let declared = owned_cases.iter().try_fold(0usize, |total, (_, _, rows)| {
        total
            .checked_add(rows.len())
            .ok_or_else(|| anyhow!("dense BF16 declared-shape count overflows"))
    })?;
    let max_shapes = u32::try_from(declared)
        .context("dense BF16 declared-shape count exceeds calibration API")?;
    let cases: Vec<_> = owned_cases
        .iter()
        .map(|(shape, weight, reachable_m)| DenseBf16CalibrationCase {
            weight: *weight,
            shape: *shape,
            reachable_m,
        })
        .collect();
    let limits = DenseBf16CalibrationLimits {
        max_shapes,
        ..DenseBf16CalibrationLimits::default()
    };
    let (plan, receipt) =
        calibrate_dense_bf16_routes(registry, device, activation_epoch, limits, &cases)
            .context("calibrate native BF16 dense routes")?;
    ensure!(
        receipt.declared_shapes == max_shapes,
        "dense BF16 receipt covered {} shapes, expected {}",
        receipt.declared_shapes,
        max_shapes
    );
    tracing::info!(
        activation_epoch,
        plan_id = plan.plan_id(),
        declared_shapes = receipt.declared_shapes,
        calibrated_decisions = receipt.calibrated_decisions,
        cache_hits = receipt.process_cache_hits,
        budget_fallbacks = receipt.budget_fallback_decisions,
        calibration_submissions = receipt.calibration_submissions,
        elapsed_ms = receipt.elapsed_ms,
        "activated native BF16 dense routes"
    );
    Ok(Some(DenseBf16Activation { plan, receipt }))
}

/// Install an ordinary calibrated plan for a focused low-level GPU test.
///
/// Production model/session entrypoints must inventory their complete matrix
/// union before becoming live. Low-level kernel tests do not own a model, so
/// they call this helper explicitly with the matrices they exercise instead
/// of bypassing auto routing or weakening its fail-closed contract.
#[cfg(test)]
pub(crate) fn activate_native_bf16_test_plan(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    matrices: &[NativeBf16Matrix<'_>],
) -> Result<()> {
    if registry.dense_bf16_plan().is_some() {
        return Ok(());
    }
    static NEXT_TEST_ACTIVATION_EPOCH: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(1);
    let epoch = NEXT_TEST_ACTIVATION_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ensure!(epoch != 0, "dense BF16 test activation epoch exhausted");
    ensure!(
        activate_native_bf16_dense(registry, device, epoch, matrices)?.is_some(),
        "dense BF16 test plan requires at least one matrix"
    );
    Ok(())
}
