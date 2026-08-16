//! Deterministic RGB8 bicubic resize used by Qwen vision preprocessing.
//!
//! The model's image processor defines a separable cubic convolution with
//! `a = -0.5`, pixel-center coordinates, widened support while downscaling,
//! normalized coefficients, and 22-bit fixed-point accumulation. Keeping the
//! implementation here avoids silently substituting a library filter whose
//! edge and rounding behavior differs from the model contract.

use image::{Rgb, RgbImage};

const PRECISION_BITS: u32 = 22;
const ROUNDING: i64 = 1_i64 << (PRECISION_BITS - 1);

#[derive(Debug)]
struct Coefficients {
    bounds: Vec<(usize, usize)>,
    weights: Vec<Vec<i32>>,
}

fn cubic_filter(mut x: f64) -> f64 {
    x = x.abs();
    if x < 1.0 {
        return (1.5 * x - 2.5) * x * x + 1.0;
    }
    if x < 2.0 {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    0.0
}

fn coefficients(input: usize, output: usize) -> Coefficients {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let mut bounds = Vec::with_capacity(output);
    let mut all_weights = Vec::with_capacity(output);

    for out in 0..output {
        let center = (out as f64 + 0.5) * scale;
        let mut first = (center - support + 0.5) as isize;
        let mut last = (center + support + 0.5) as isize;
        first = first.max(0);
        last = last.min(input as isize);
        let first = first as usize;
        let last = last.max(first as isize) as usize;
        let mut weights = Vec::with_capacity(last - first);
        let mut sum = 0.0;
        for sample in first..last {
            let weight = cubic_filter((sample as f64 - center + 0.5) / filter_scale);
            weights.push(weight);
            sum += weight;
        }
        let fixed = weights
            .into_iter()
            .map(|weight| {
                let normalized = if sum == 0.0 { 0.0 } else { weight / sum };
                let scaled = normalized * ((1_u64 << PRECISION_BITS) as f64);
                if scaled < 0.0 {
                    (-0.5 + scaled) as i32
                } else {
                    (0.5 + scaled) as i32
                }
            })
            .collect();
        bounds.push((first, last));
        all_weights.push(fixed);
    }

    Coefficients {
        bounds,
        weights: all_weights,
    }
}

fn clip_accumulator(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

fn resize_horizontal(source: &RgbImage, target_w: u32) -> RgbImage {
    if source.width() == target_w {
        return source.clone();
    }
    let coeffs = coefficients(source.width() as usize, target_w as usize);
    let mut output = RgbImage::new(target_w, source.height());
    for y in 0..source.height() {
        for x in 0..target_w as usize {
            let (first, last) = coeffs.bounds[x];
            let mut sums = [ROUNDING; 3];
            for (sample, &weight) in (first..last).zip(&coeffs.weights[x]) {
                let pixel = source.get_pixel(sample as u32, y);
                for channel in 0..3 {
                    sums[channel] += i64::from(pixel[channel]) * i64::from(weight);
                }
            }
            output.put_pixel(
                x as u32,
                y,
                Rgb([
                    clip_accumulator(sums[0]),
                    clip_accumulator(sums[1]),
                    clip_accumulator(sums[2]),
                ]),
            );
        }
    }
    output
}

/// Resize an RGB8 image using the Qwen processor's exact bicubic contract.
pub(super) fn resize_rgb8(source: &RgbImage, target_w: u32, target_h: u32) -> RgbImage {
    let horizontal = resize_horizontal(source, target_w);
    if horizontal.height() == target_h {
        return horizontal;
    }
    let coeffs = coefficients(horizontal.height() as usize, target_h as usize);
    let mut output = RgbImage::new(target_w, target_h);
    for y in 0..target_h as usize {
        let (first, last) = coeffs.bounds[y];
        for x in 0..target_w {
            let mut sums = [ROUNDING; 3];
            for (sample, &weight) in (first..last).zip(&coeffs.weights[y]) {
                let pixel = horizontal.get_pixel(x, sample as u32);
                for channel in 0..3 {
                    sums[channel] += i64::from(pixel[channel]) * i64::from(weight);
                }
            }
            output.put_pixel(
                x,
                y as u32,
                Rgb([
                    clip_accumulator(sums[0]),
                    clip_accumulator(sums[1]),
                    clip_accumulator(sums[2]),
                ]),
            );
        }
    }
    output
}
