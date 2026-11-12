//! Official-artifact gate for the prepared routes of a Qwen3.6 DFlash2
//! target verify step at every tile width
//! (`HF2Q_REPACKED_GGUF_ANY_WIDTH=1` with the prepared FFN-free projections
//! and head): a lane's rows must be bitwise the same whether its step is an
//! 8-, 16- or 32-row tile, whichever lanes share it, with padding rows
//! beside it, on the host-token and the GPU-token (route mask) paths, with
//! the per-row and the tiled verify attention.
//!
//! The prepared-route switches are set by the test itself
//! ([`prepared_switch_override`](super::prepared_switch_override)); the MoE
//! route is chosen at model upload, so it comes from the environment:
//!
//! ```text
//! HF2Q_QWEN36_GGUF=/path/qwen36.gguf HF2Q_QWEN_MOE_BATCHED=1 \
//! HF2Q_QWEN_MOE_STAGED_M8=1 HF2Q_QWEN_MOE_LIVE_TILES=1 \
//!   cargo test --release --bin hf2q \
//!   qwen36_dflash2_verify_rows_have_one_value_at_every_tile_width -- --ignored
//! ```
//!
//! Every window here starts at the same cursor. A token verified at another
//! window offset is outside this gate; for the tiled attention that is
//! `verify_attn_tile_window_tests` (bitwise equal from mlx-native 0.21.9).

use mlx_native::DType;

use super::forward_gpu::Qwen35JointVerifyLane;
use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::joint_projection::JointProjectionKernel;
use super::joint_step_options::JointStepOptions;
use super::model::Qwen35Model;
use super::prepared_switch_override::{Override, PreparedSwitches};
use super::real_artifact_joint_decode_invariance_tests::cache_with_capacity;
use super::real_artifact_joint_decode_tests::{
    argmax, bits, hidden_bits, load, prompt, GateRouting, SLOTS,
};
use super::spec_decode::positions_for_range;
use super::Qwen35JointOutput;
use crate::inference::spec_decode::dflash2::adapters::DFlash2TargetCapture;
use crate::serve::multi_seq_kv::SlotId;

const TARGET: u32 = 2;
const CAPACITY: u32 = 160;
/// Draft tokens of a window after its pending token.
const DRAFTS: [u32; 7] = [374, 279, 15, 1_782, 9_707, 220, 13];

/// The target lane's rows: logits, post-norm hidden, every layer's output.
struct Trace {
    logits: Vec<Vec<u32>>,
    hidden: Vec<Vec<u32>>,
    layers: Vec<Vec<u32>>,
}

/// One DFlash2 verify of `cohort` (`(slot, rows)` in packed order) after an
/// identical prefill of its slots. `live` gives every physical lane's
/// GPU-selected live rows and selects the GPU-token path.
fn verify(
    model: &Qwen35Model,
    cohort: &[(u32, usize)],
    tile: bool,
    live: Option<[u32; 4]>,
) -> Trace {
    let h = model.cfg.hidden_size as usize;
    let n_layers = model.cfg.num_hidden_layers as usize;
    let mut cache = cache_with_capacity(model, CAPACITY);
    let seeds = cohort
        .iter()
        .map(|&(slot, _)| {
            let tokens = prompt(slot, 0);
            let logits = model
                .forward_gpu_last_logits(
                    &tokens,
                    &positions_for_range(0, tokens.len()),
                    &mut cache,
                    SlotId(slot),
                )
                .expect("prefill");
            (argmax(&logits), tokens.len() as i32)
        })
        .collect::<Vec<_>>();
    let lane_tokens = cohort
        .iter()
        .zip(&seeds)
        .map(|(&(slot, rows), &(pending, _))| {
            let mut tokens = vec![pending];
            tokens.extend((1..rows).map(|row| DRAFTS[(row - 1 + slot as usize) % DRAFTS.len()]));
            tokens
        })
        .collect::<Vec<_>>();
    let lanes = cohort
        .iter()
        .zip(&seeds)
        .zip(&lane_tokens)
        .map(
            |((&(slot, _), &(_, position)), tokens)| Qwen35JointVerifyLane {
                slot_id: SlotId(slot),
                position,
                tokens,
                output: Qwen35JointOutput::LogitsAndHidden,
            },
        )
        .collect::<Vec<_>>();
    let layer_ids = (0..n_layers).collect::<Vec<_>>();
    let feature_width = h * n_layers;
    let (captured, verify_rows, token_tile) = model
        .with_gpu_cache_mut(|device, _registry| {
            let captured = device.alloc_buffer(
                SLOTS as usize * 8 * feature_width * 4,
                DType::F32,
                vec![SLOTS as usize * 8, feature_width],
            )?;
            let mut verify_rows =
                device.alloc_buffer(SLOTS as usize * 4, DType::U32, vec![SLOTS as usize])?;
            verify_rows
                .as_mut_slice::<u32>()?
                .copy_from_slice(&live.unwrap_or([8; 4]));
            let mut token_tile = device.alloc_buffer(32 * 4, DType::U32, vec![32])?;
            let flat = lane_tokens.iter().flatten().copied().collect::<Vec<_>>();
            for (index, token) in token_tile.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                *token = flat[if index < flat.len() { index } else { 0 }];
            }
            Ok((captured, verify_rows, token_tile))
        })
        .expect("allocate DFlash2 capture");
    let capture_lanes = cohort
        .iter()
        .map(|&(slot, _)| slot as usize)
        .collect::<Vec<_>>();
    let capture = DFlash2TargetCapture {
        layer_ids: &layer_ids,
        output: &captured,
        lane_indices: &capture_lanes,
        verify_rows: &verify_rows,
        lanes: SLOTS as usize,
        rows_per_lane: 8,
        feature_width,
    };
    let options = JointStepOptions {
        projection: JointProjectionKernel::Canonical,
        pooled_outputs: true,
        verify_attention_tile: tile,
    };
    let verify = if live.is_some() {
        model.forward_gpu_dflash2_verify_from_gpu_tokens(
            &lanes,
            &token_tile,
            &mut cache,
            JointRouteMode::Exact,
            JointDenseFfnRoute::SerialBuilder,
            options,
            Some(capture),
            None,
        )
    } else {
        model.forward_gpu_dflash2_verify(
            &lanes,
            &mut cache,
            JointRouteMode::Exact,
            JointDenseFfnRoute::SerialBuilder,
            options,
            Some(capture),
        )
    }
    .unwrap_or_else(|error| panic!("DFlash2 verify of {cohort:?}: {error:#}"));
    let index = cohort
        .iter()
        .position(|&(slot, _)| slot == TARGET)
        .expect("the cohort holds the target lane");
    let rows = &verify.rows[index];
    let captured = captured.as_slice::<f32>().expect("read capture");
    let start = TARGET as usize * 8 * feature_width;
    Trace {
        logits: rows
            .iter()
            .map(|row| bits(row.logits.as_ref().expect("logits")))
            .collect(),
        hidden: rows
            .iter()
            .map(|row| hidden_bits(row.nextn_hidden.as_ref().expect("hidden"), h))
            .collect(),
        layers: (0..rows.len())
            .map(|row| {
                bits(&captured[start + row * feature_width..start + (row + 1) * feature_width])
            })
            .collect(),
    }
}

/// The first `rows` rows of `trace` are those of `reference`, or the label
/// of the first difference: its row and the first layer whose output moved.
fn first_difference(
    model: &Qwen35Model,
    trace: &Trace,
    reference: &Trace,
    rows: usize,
) -> Option<String> {
    let h = model.cfg.hidden_size as usize;
    (0..rows).find_map(|row| {
        if trace.logits[row] == reference.logits[row] && trace.hidden[row] == reference.hidden[row]
        {
            return None;
        }
        let layer = trace.layers[row]
            .chunks_exact(h)
            .zip(reference.layers[row].chunks_exact(h))
            .position(|(a, b)| a != b);
        Some(match layer {
            Some(layer) => format!(
                "row {row} from layer {layer} ({:?})",
                model.cfg.layer_types[layer]
            ),
            None => format!("row {row} in the final norm or head"),
        })
    })
}

#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact), the MoE route switches and Apple Metal"]
fn qwen36_dflash2_verify_rows_have_one_value_at_every_tile_width() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN36_GGUF", GateRouting::Exact);
    assert_eq!(model.joint_decode_mode(), Ok(JointRouteMode::Exact));
    println!(
        "Qwen3.6 MoE route {} / {}",
        model.moe_ffn_decode_route(),
        model.moe_expert_decode_route()
    );
    let stock_8 = {
        let _off = Override::set(PreparedSwitches::OFF);
        verify(&model, &[(TARGET, 8)], false, None)
    };
    let _switches = Override::set(PreparedSwitches::ALL);
    let mut failures = Vec::new();
    for tile in [false, true] {
        let attention = if tile {
            "tiled attention"
        } else {
            "per-row attention"
        };
        let reference = verify(&model, &[(TARGET, 8)], tile, None);
        if !tile {
            assert!(
                reference.logits != stock_8.logits,
                "the 8-row tile kept the stock kernels"
            );
        }
        let host: [(&str, &[(u32, usize)], usize); 10] = [
            ("8 rows alone, repeat", &[(TARGET, 8)], 8),
            ("width 16, target last", &[(0, 8), (TARGET, 8)], 8),
            ("width 16, target first", &[(TARGET, 8), (3, 8)], 8),
            ("width 16, 11 live rows", &[(0, 3), (TARGET, 8)], 8),
            ("width 16, 9 live rows", &[(TARGET, 8), (1, 1)], 8),
            ("width 32, 24 live rows", &[(0, 8), (1, 8), (TARGET, 8)], 8),
            ("width 32, full", &[(0, 8), (1, 8), (TARGET, 8), (3, 8)], 8),
            (
                "width 32, full, target first",
                &[(TARGET, 8), (0, 8), (1, 8), (3, 8)],
                8,
            ),
            ("width 4, 4-row window", &[(TARGET, 4)], 4),
            ("width 16, 4-row window", &[(0, 8), (TARGET, 4)], 4),
        ];
        for (label, cohort, rows) in host {
            let trace = verify(&model, cohort, tile, None);
            if let Some(difference) = first_difference(&model, &trace, &reference, rows) {
                failures.push(format!("{attention}, host tokens, {label}: {difference}"));
            }
        }
        let gpu: [(&str, &[(u32, usize)], [u32; 4], usize); 7] = [
            ("8 rows alone", &[(TARGET, 8)], [8, 8, 8, 8], 8),
            ("width 16", &[(0, 8), (TARGET, 8)], [8, 8, 8, 8], 8),
            (
                "width 16, other lane 3 live rows",
                &[(0, 8), (TARGET, 8)],
                [3, 8, 8, 8],
                8,
            ),
            (
                "width 32, 24 live rows",
                &[(0, 8), (1, 8), (TARGET, 8)],
                [8, 8, 8, 8],
                8,
            ),
            (
                "width 32, full",
                &[(0, 8), (1, 8), (TARGET, 8), (3, 8)],
                [8, 8, 8, 8],
                8,
            ),
            (
                "width 32, other lanes 1 live row",
                &[(0, 8), (1, 8), (TARGET, 8), (3, 8)],
                [1, 1, 8, 1],
                8,
            ),
            (
                "width 32, target 4 live rows",
                &[(0, 8), (1, 8), (TARGET, 8), (3, 8)],
                [8, 8, 4, 8],
                4,
            ),
        ];
        for (label, cohort, live, rows) in gpu {
            let trace = verify(&model, cohort, tile, Some(live));
            if let Some(difference) = first_difference(&model, &trace, &reference, rows) {
                failures.push(format!("{attention}, GPU tokens, {label}: {difference}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "a lane's rows depend on its step:\n{}",
        failures.join("\n")
    );
}
