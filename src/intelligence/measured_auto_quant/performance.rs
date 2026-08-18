use std::collections::BTreeSet;

use super::{
    reject, valid_nonnegative, valid_sha256, CandidateReceipt, CandidateRejectionCode,
    CandidateRejectionReason, PerformanceMeasurement, SelectionContract,
};

pub(super) fn check_performance(
    contract: &SelectionContract,
    candidate: &CandidateReceipt,
    reasons: &mut Vec<CandidateRejectionReason>,
) {
    let mut seen = BTreeSet::new();
    for measurement in &candidate.performance {
        if !seen.insert(measurement.regime) {
            reject(
                reasons,
                CandidateRejectionCode::DuplicatePerformanceMeasurement,
                format!(
                    "duplicate performance measurement: {:?}",
                    measurement.regime
                ),
            );
        }
        if !valid_sha256(&measurement.workload_sha256)
            || !valid_nonnegative(measurement.median_tokens_per_second)
            || !valid_nonnegative(measurement.median_semantic_ttft_ms)
            || measurement.peak_mlx_bytes == 0
            || measurement.measured_runs == 0
            || measurement.tokens_per_run == 0
            || !measurement.output_quality_cases.is_complete()
        {
            reject(
                reasons,
                CandidateRejectionCode::InvalidPerformanceMeasurement,
                format!(
                    "invalid performance measurement for {:?}",
                    measurement.regime
                ),
            );
        }
    }

    for requirement in &contract.required_regimes {
        let Some(measurement) = candidate
            .performance
            .iter()
            .find(|measurement| measurement.regime == requirement.regime)
        else {
            reject(
                reasons,
                CandidateRejectionCode::MissingPerformanceRegime,
                format!(
                    "required performance regime is missing: {:?}",
                    requirement.regime
                ),
            );
            continue;
        };

        if !valid_nonnegative(measurement.median_tokens_per_second)
            || measurement.median_tokens_per_second < requirement.min_tokens_per_second
        {
            reject(
                reasons,
                CandidateRejectionCode::ThroughputGateFailed,
                format!("throughput gate failed for {:?}", requirement.regime),
            );
        }
        if measurement.workload_sha256 != requirement.workload_sha256 {
            reject(
                reasons,
                CandidateRejectionCode::WorkloadIdentityMismatch,
                format!("workload identity mismatch for {:?}", requirement.regime),
            );
        }
        if let Some(max_ttft_ms) = requirement.max_semantic_ttft_ms {
            if !valid_nonnegative(measurement.median_semantic_ttft_ms)
                || measurement.median_semantic_ttft_ms > max_ttft_ms
            {
                reject(
                    reasons,
                    CandidateRejectionCode::SemanticTtftGateFailed,
                    format!("semantic TTFT gate failed for {:?}", requirement.regime),
                );
            }
        }
        if measurement.peak_mlx_bytes > contract.max_peak_mlx_bytes {
            reject(
                reasons,
                CandidateRejectionCode::MemoryGateFailed,
                format!("memory gate failed for {:?}", requirement.regime),
            );
        }
        if measurement.warmup_runs < requirement.min_warmup_runs
            || measurement.measured_runs < requirement.min_measured_runs
            || measurement.tokens_per_run < requirement.min_tokens_per_run
        {
            reject(
                reasons,
                CandidateRejectionCode::InsufficientBenchmarkEvidence,
                format!(
                    "insufficient benchmark evidence for {:?}",
                    requirement.regime
                ),
            );
        }
    }
}

pub(super) fn compare_candidates(
    contract: &SelectionContract,
    left: &CandidateReceipt,
    right: &CandidateReceipt,
) -> std::cmp::Ordering {
    let left_primary = primary_measurement(contract, left);
    let right_primary = primary_measurement(contract, right);

    right_primary
        .median_tokens_per_second
        .total_cmp(&left_primary.median_tokens_per_second)
        .then_with(|| {
            left_primary
                .median_semantic_ttft_ms
                .total_cmp(&right_primary.median_semantic_ttft_ms)
        })
        .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
        .then_with(|| left.recipe.candidate_id.cmp(&right.recipe.candidate_id))
}

fn primary_measurement<'a>(
    contract: &SelectionContract,
    candidate: &'a CandidateReceipt,
) -> &'a PerformanceMeasurement {
    // Eligibility already proved this measurement exists exactly once.
    candidate
        .performance
        .iter()
        .find(|measurement| measurement.regime == contract.primary_regime)
        .expect("eligible candidate must contain primary performance evidence")
}
