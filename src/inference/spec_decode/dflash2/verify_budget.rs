//! Online verification-row budget for MoE DFlash2 targets.
//!
//! The model is source-bound to Splash `031a25a6`: proposal confidence is
//! calibrated by MAP logistic regression, while recent target-cycle wall times
//! fit a row-cost line. Dense targets do not use this budget.

use anyhow::{ensure, Result};

use super::workspace::DFlash2Workspace;
use mlx_native::ops::dflash2_selector::{
    DFLASH2_MAX_LANES, DFLASH2_MAX_POSITIONS, DFLASH2_VERIFY_COSTS,
};

const POSITIONS: usize = 7;
const BLOCK_ROWS: usize = 8;
const PARAMETERS: usize = 1 + POSITIONS;
const REFIT_CYCLES: u64 = 64;
const MAX_LABELS: usize = 4096;
const EXPLORE_CYCLES: u64 = 32;
const DECAY: f64 = 0.97;
const PRIOR_CYCLES: f64 = 5.0;
const SLOPE_PRECISION: f64 = 50.0;
const INTERCEPT_PRECISION: f64 = 4.0;
const FIT_ITERATIONS: usize = 16;

const COST_KNOTS: [f64; 4] = [1.0, 2.0, 4.0, 8.0];
const COST_SHAPES: [[f64; 4]; DFLASH2_MAX_LANES] = [
    [0.60, 0.69, 0.85, 1.0],
    [0.50, 0.58, 0.74, 1.0],
    [0.45, 0.52, 0.70, 1.0],
    [0.40, 0.47, 0.67, 1.0],
];

#[derive(Clone, Copy, Debug)]
struct Acceptance {
    slope: f32,
    intercept: [f32; POSITIONS],
}

const PRIOR: Acceptance = Acceptance {
    slope: 0.7,
    intercept: [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45],
};

#[derive(Clone, Copy, Debug)]
struct Label {
    position: usize,
    feature: f32,
    accepted: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    weight: f64,
    rows: f64,
    rows_squared: f64,
    time: f64,
    rows_time: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VerifyBudgetLane {
    pub lane_index: usize,
    pub verify_rows: usize,
    pub accepted: usize,
    pub labels: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct VerifyRowsModel {
    pub slope: f32,
    pub intercept: [f32; DFLASH2_MAX_POSITIONS],
    pub costs: [f32; DFLASH2_VERIFY_COSTS],
    pub every_row: bool,
}

#[derive(Debug)]
pub(crate) struct DFlash2VerifyBudget {
    acceptance: Acceptance,
    labels: Vec<Label>,
    next_label: usize,
    cycles: u64,
    costs: [Cost; DFLASH2_MAX_LANES],
    table_cycles: [u64; DFLASH2_MAX_LANES],
}

impl Default for DFlash2VerifyBudget {
    fn default() -> Self {
        Self {
            acceptance: PRIOR,
            labels: Vec::with_capacity(MAX_LABELS),
            next_label: 0,
            cycles: 0,
            costs: [Cost::default(); DFLASH2_MAX_LANES],
            table_cycles: [0; DFLASH2_MAX_LANES],
        }
    }
}

impl DFlash2VerifyBudget {
    pub(crate) fn model(&self, lanes: usize) -> Result<VerifyRowsModel> {
        ensure!(
            (1..=DFLASH2_MAX_LANES).contains(&lanes),
            "invalid DFlash2 verify-budget lane count {lanes}"
        );
        let mut intercept = [0.0; DFLASH2_MAX_POSITIONS];
        intercept[..POSITIONS].copy_from_slice(&self.acceptance.intercept);
        Ok(VerifyRowsModel {
            slope: self.acceptance.slope,
            intercept,
            costs: self.modeled_costs(lanes),
            every_row: self.table_cycles[lanes - 1] % EXPLORE_CYCLES == EXPLORE_CYCLES - 1,
        })
    }

    pub(crate) fn select_rows(
        &self,
        workspace: &DFlash2Workspace,
        lane_indices: &[usize],
        maximum_rows: &[u32],
    ) -> Result<[usize; DFLASH2_MAX_LANES]> {
        ensure!(
            !lane_indices.is_empty() && lane_indices.len() <= DFLASH2_MAX_LANES,
            "invalid DFlash2 verify-budget selection lane count {}",
            lane_indices.len()
        );
        ensure!(
            maximum_rows.len() == workspace.max_lanes,
            "DFlash2 maximum-row count {} != workspace lanes {}",
            maximum_rows.len(),
            workspace.max_lanes
        );
        let all_confidences = workspace.confidence.as_slice::<f32>()?;
        ensure!(
            all_confidences.len() == workspace.max_lanes * POSITIONS,
            "DFlash2 confidence count {} != {}",
            all_confidences.len(),
            workspace.max_lanes * POSITIONS
        );
        let mut confidences = [0.0; DFLASH2_MAX_LANES * POSITIONS];
        let mut maximum = [1; DFLASH2_MAX_LANES];
        for (position, &lane) in lane_indices.iter().enumerate() {
            ensure!(
                lane < workspace.max_lanes && !lane_indices[..position].contains(&lane),
                "invalid or repeated DFlash2 verify-budget lane {lane}"
            );
            let rows = maximum_rows[lane] as usize;
            ensure!(
                (1..=BLOCK_ROWS).contains(&rows),
                "invalid DFlash2 maximum verify rows {rows} for lane {lane}"
            );
            confidences[position * POSITIONS..(position + 1) * POSITIONS]
                .copy_from_slice(&all_confidences[lane * POSITIONS..(lane + 1) * POSITIONS]);
            maximum[position] = rows;
        }
        select_rows_from_confidence(
            &self.model(lane_indices.len())?,
            &confidences[..lane_indices.len() * POSITIONS],
            &maximum[..lane_indices.len()],
        )
    }

    pub(crate) fn observe(
        &mut self,
        workspace: &DFlash2Workspace,
        lanes: &[VerifyBudgetLane],
        milliseconds: f64,
    ) -> Result<()> {
        ensure!(
            !lanes.is_empty() && lanes.len() <= DFLASH2_MAX_LANES,
            "invalid DFlash2 verify-budget observation lane count {}",
            lanes.len()
        );
        let confidences = workspace.confidence.as_slice::<f32>()?;
        let expected = workspace.max_lanes * POSITIONS;
        ensure!(
            confidences.len() == expected,
            "DFlash2 confidence count {} != {expected}",
            confidences.len()
        );

        let mut rows = 0usize;
        for lane in lanes {
            ensure!(
                lane.lane_index < workspace.max_lanes,
                "DFlash2 verify-budget lane {} outside {}",
                lane.lane_index,
                workspace.max_lanes
            );
            ensure!(
                (1..=BLOCK_ROWS).contains(&lane.verify_rows),
                "invalid DFlash2 verify-row count {}",
                lane.verify_rows
            );
            ensure!(
                lane.accepted <= POSITIONS,
                "invalid DFlash2 accepted proposal count {}",
                lane.accepted
            );
            rows += lane.verify_rows;
        }

        let lane_count = lanes.len();
        self.table_cycles[lane_count - 1] += 1;
        if milliseconds.is_finite() && milliseconds > 0.0 {
            let mut observed = milliseconds;
            if let Some((intercept, slope)) = self.cost_line(lane_count) {
                let expected = (intercept + slope * rows as f64).max(1.0e-3);
                observed = observed.clamp(0.5 * expected, 2.0 * expected);
            }
            let cost = &mut self.costs[lane_count - 1];
            cost.weight = DECAY * cost.weight + 1.0;
            cost.rows = DECAY * cost.rows + rows as f64;
            cost.rows_squared = DECAY * cost.rows_squared + (rows * rows) as f64;
            cost.time = DECAY * cost.time + observed;
            cost.rows_time = DECAY * cost.rows_time + rows as f64 * observed;
        }

        for lane in lanes.iter().filter(|lane| lane.labels) {
            let last = lane.accepted.min(lane.verify_rows - 1).min(POSITIONS - 1);
            let start = lane.lane_index * POSITIONS;
            for position in 0..=last {
                let confidence = confidences[start + position];
                if !(0.0..=1.0).contains(&confidence) {
                    continue;
                }
                self.push_label(Label {
                    position,
                    feature: logit(confidence) as f32,
                    accepted: position < lane.accepted,
                });
            }
        }
        self.cycles += 1;
        if self.cycles % REFIT_CYCLES == 0 && !self.labels.is_empty() {
            self.acceptance = fit_acceptance(&self.labels, self.acceptance);
        }
        Ok(())
    }

    fn push_label(&mut self, label: Label) {
        if self.labels.len() < MAX_LABELS {
            self.labels.push(label);
        } else {
            self.labels[self.next_label] = label;
            self.next_label = (self.next_label + 1) % MAX_LABELS;
        }
    }

    fn modeled_costs(&self, lanes: usize) -> [f32; DFLASH2_VERIFY_COSTS] {
        let mut result = [1.0; DFLASH2_VERIFY_COSTS];
        let first = lanes;
        let last = lanes * BLOCK_ROWS;
        if let Some((intercept, slope)) = self.cost_line(lanes) {
            let mean = self.costs[lanes - 1].time / self.costs[lanes - 1].weight;
            for (rows, cost) in result.iter_mut().enumerate().take(last + 1).skip(first) {
                *cost = (intercept + slope * rows as f64).max(0.05 * mean) as f32;
            }
        } else {
            for (rows, cost) in result.iter_mut().enumerate().take(last + 1).skip(first) {
                *cost = seeded_cost(lanes, rows) as f32;
            }
        }
        result
    }

    fn cost_line(&self, lanes: usize) -> Option<(f64, f64)> {
        let cost = self.costs[lanes - 1];
        if cost.weight <= 0.0 {
            return None;
        }
        let first = lanes;
        let last = lanes * BLOCK_ROWS;
        let mean_rows = cost.rows / cost.weight;
        let mean_time = cost.time / cost.weight;
        let spread = (cost.rows_squared / cost.weight - mean_rows * mean_rows).max(0.0);
        let covariance = cost.rows_time / cost.weight - mean_rows * mean_time;
        let seed_slope =
            (seeded_cost(lanes, last) - seeded_cost(lanes, first)) / (last - first) as f64;
        let prior_slope = seed_slope * mean_time / seeded_cost(lanes, mean_rows.round() as usize);
        let prior_spread = (lanes * lanes) as f64;
        let slope = ((cost.weight * covariance + PRIOR_CYCLES * prior_spread * prior_slope)
            / (cost.weight * spread + PRIOR_CYCLES * prior_spread))
            .max(0.0);
        Some((mean_time - slope * mean_rows, slope))
    }
}

fn seeded_cost(lanes: usize, rows: usize) -> f64 {
    debug_assert!((1..=DFLASH2_MAX_LANES).contains(&lanes));
    debug_assert!((lanes..=lanes * BLOCK_ROWS).contains(&rows));
    let per_lane = rows as f64 / lanes as f64;
    let upper = if per_lane <= 2.0 {
        1
    } else if per_lane <= 4.0 {
        2
    } else {
        3
    };
    let t = (per_lane - COST_KNOTS[upper - 1]) / (COST_KNOTS[upper] - COST_KNOTS[upper - 1]);
    let shape = COST_SHAPES[lanes - 1];
    shape[upper - 1] + t * (shape[upper] - shape[upper - 1])
}

fn select_rows_from_confidence(
    model: &VerifyRowsModel,
    confidences: &[f32],
    maximum: &[usize],
) -> Result<[usize; DFLASH2_MAX_LANES]> {
    let lanes = maximum.len();
    ensure!(
        !maximum.is_empty()
            && maximum.len() <= DFLASH2_MAX_LANES
            && confidences.len() == lanes * POSITIONS,
        "invalid DFlash2 host verify-row geometry"
    );
    let mut rows = [1; DFLASH2_MAX_LANES];
    if model.every_row {
        rows[..lanes].copy_from_slice(maximum);
        return Ok(rows);
    }
    let mut ranked = [(0.0_f32, 0_usize); DFLASH2_MAX_LANES * POSITIONS];
    let mut ranked_count = 0;
    for lane in 0..lanes {
        let mut survival = 1.0_f32;
        for position in 0..maximum[lane].saturating_sub(1) {
            let confidence = confidences[lane * POSITIONS + position].clamp(1.0e-6, 1.0 - 1.0e-6);
            let logit = (confidence / (1.0 - confidence)).ln();
            let z = model.slope * logit + model.intercept[position];
            survival *= 1.0 / (1.0 + (-z).exp());
            let mut at = ranked_count;
            while at > 0 && ranked[at - 1].0 < survival {
                at -= 1;
            }
            for index in (at..ranked_count).rev() {
                ranked[index + 1] = ranked[index];
            }
            ranked[at] = (survival, lane);
            ranked_count += 1;
        }
    }
    let mut expected = lanes as f32;
    let mut best = expected / model.costs[lanes];
    let mut taken = 0;
    for (index, (survival, _)) in ranked[..ranked_count].iter().enumerate() {
        expected += survival;
        let value = expected / model.costs[lanes + index + 1];
        if value > best {
            best = value;
            taken = index + 1;
        }
    }
    for &(_, lane) in &ranked[..taken] {
        rows[lane] += 1;
    }
    Ok(rows)
}

fn logit(probability: f32) -> f64 {
    let probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6) as f64;
    (probability / (1.0 - probability)).ln()
}

fn parameters(acceptance: Acceptance) -> [f64; PARAMETERS] {
    let mut result = [0.0; PARAMETERS];
    result[0] = acceptance.slope as f64;
    for (position, value) in acceptance.intercept.into_iter().enumerate() {
        result[1 + position] = value as f64;
    }
    result
}

fn objective(labels: &[Label], theta: &[f64; PARAMETERS]) -> f64 {
    let prior = parameters(PRIOR);
    let mut value = 0.0;
    for label in labels {
        let z = theta[0] * label.feature as f64 + theta[1 + label.position];
        let margin = if label.accepted { z } else { -z };
        value += if margin > 0.0 {
            (-margin).exp().ln_1p()
        } else {
            -margin + margin.exp().ln_1p()
        };
    }
    for index in 0..PARAMETERS {
        let precision = if index == 0 {
            SLOPE_PRECISION
        } else {
            INTERCEPT_PRECISION
        };
        value += 0.5 * precision * (theta[index] - prior[index]).powi(2);
    }
    value
}

fn solve(
    mut hessian: [[f64; PARAMETERS]; PARAMETERS],
    mut gradient: [f64; PARAMETERS],
) -> [f64; PARAMETERS] {
    for column in 0..PARAMETERS {
        let mut pivot = column;
        for row in column + 1..PARAMETERS {
            if hessian[row][column].abs() > hessian[pivot][column].abs() {
                pivot = row;
            }
        }
        hessian.swap(column, pivot);
        gradient.swap(column, pivot);
        for row in column + 1..PARAMETERS {
            let factor = hessian[row][column] / hessian[column][column];
            for index in column..PARAMETERS {
                hessian[row][index] -= factor * hessian[column][index];
            }
            gradient[row] -= factor * gradient[column];
        }
    }
    let mut result = [0.0; PARAMETERS];
    for row in (0..PARAMETERS).rev() {
        let mut sum = gradient[row];
        for index in row + 1..PARAMETERS {
            sum -= hessian[row][index] * result[index];
        }
        result[row] = sum / hessian[row][row];
    }
    result
}

fn fit_acceptance(labels: &[Label], start: Acceptance) -> Acceptance {
    let prior = parameters(PRIOR);
    let mut theta = parameters(start);
    let mut current = objective(labels, &theta);
    for _ in 0..FIT_ITERATIONS {
        let mut gradient = [0.0; PARAMETERS];
        let mut hessian = [[0.0; PARAMETERS]; PARAMETERS];
        for label in labels {
            let intercept = 1 + label.position;
            let feature = label.feature as f64;
            let probability = 1.0 / (1.0 + (-(theta[0] * feature + theta[intercept])).exp());
            let residual = probability - if label.accepted { 1.0 } else { 0.0 };
            let weight = probability * (1.0 - probability);
            gradient[0] += residual * feature;
            gradient[intercept] += residual;
            hessian[0][0] += weight * feature * feature;
            hessian[0][intercept] += weight * feature;
            hessian[intercept][0] += weight * feature;
            hessian[intercept][intercept] += weight;
        }
        for index in 0..PARAMETERS {
            let precision = if index == 0 {
                SLOPE_PRECISION
            } else {
                INTERCEPT_PRECISION
            };
            gradient[index] += precision * (theta[index] - prior[index]);
            hessian[index][index] += precision;
        }
        let step = solve(hessian, gradient);
        let mut next = theta;
        let mut next_value = current;
        let mut scale = 1.0;
        let mut moved = false;
        for _ in 0..30 {
            for index in 0..PARAMETERS {
                next[index] = theta[index] - scale * step[index];
            }
            next_value = objective(labels, &next);
            if next_value <= current {
                moved = true;
                break;
            }
            scale *= 0.5;
        }
        if !moved {
            break;
        }
        let largest = next
            .iter()
            .zip(theta)
            .map(|(next, prior)| (next - prior).abs())
            .fold(0.0, f64::max);
        theta = next;
        current = next_value;
        if largest < 1.0e-9 {
            break;
        }
    }
    let mut intercept = [0.0; POSITIONS];
    for position in 0..POSITIONS {
        intercept[position] = theta[1 + position] as f32;
    }
    Acceptance {
        slope: theta[0] as f32,
        intercept,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_costs_match_source_knots_for_every_lane_count() {
        for lanes in 1..=DFLASH2_MAX_LANES {
            for (index, rows_per_lane) in [1, 2, 4, 8].into_iter().enumerate() {
                assert_eq!(
                    seeded_cost(lanes, lanes * rows_per_lane),
                    COST_SHAPES[lanes - 1][index]
                );
            }
            assert!((lanes..=lanes * BLOCK_ROWS)
                .map(|rows| seeded_cost(lanes, rows))
                .collect::<Vec<_>>()
                .windows(2)
                .all(|pair| pair[0] <= pair[1]));
        }
    }

    #[test]
    fn repeated_rejections_lower_high_confidence_acceptance() {
        let labels = (0..256)
            .map(|_| Label {
                position: 0,
                feature: logit(0.99) as f32,
                accepted: false,
            })
            .collect::<Vec<_>>();
        let fitted = fit_acceptance(&labels, PRIOR);
        let prior_logit = PRIOR.slope * logit(0.99) as f32 + PRIOR.intercept[0];
        let fitted_logit = fitted.slope * logit(0.99) as f32 + fitted.intercept[0];
        assert!(fitted_logit < prior_logit - 1.0);
    }

    #[test]
    fn exploration_is_scheduled_after_thirty_one_cycles() {
        let mut budget = DFlash2VerifyBudget::default();
        budget.table_cycles[3] = 31;
        assert!(budget.model(4).unwrap().every_row);
        budget.table_cycles[3] = 32;
        assert!(!budget.model(4).unwrap().every_row);
    }

    #[test]
    fn confidence_budget_selects_prefixes_and_spends_more_on_stronger_paths() {
        let model = DFlash2VerifyBudget::default().model(4).unwrap();
        let maximum = [8; 4];
        let low = select_rows_from_confidence(&model, &[0.5; 28], &maximum).unwrap();
        let high = select_rows_from_confidence(&model, &[0.99; 28], &maximum).unwrap();
        assert!(low.iter().all(|rows| (1..=8).contains(rows)));
        assert!(high.iter().all(|rows| (1..=8).contains(rows)));
        assert!(low.iter().sum::<usize>() < high.iter().sum::<usize>());
    }
}
