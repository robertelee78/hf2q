//! Differential tests of the verify decision candidates against the host
//! sampler (`sampler_pure::sample_token` at temperature zero).
//!
//! Every case runs the kernels on the GPU over a `[32, vocab]` logits tile
//! and compares, row by row:
//!
//! - the token the candidates decide with the token the host sampler
//!   returns for the row's own logits and history, under several penalties
//!   and with the adversarial structure a penalty creates (a history token
//!   at the top, exact ties between a penalised history logit and the best
//!   other logit at a lower and at a higher token id, ties among other
//!   tokens, signed zeros, all-negative rows, duplicate history tokens,
//!   histories shorter and longer than the window);
//! - the logits a fallback row hands the host sampler with the head's own,
//!   bit for bit, and the tile outside the masked tokens with the head's;
//! - rows without a history with untouched logits.
//!
//! Rows the candidates must refuse (a non-finite logit inside or outside
//! the history, a history other than the one the kernel assembled) return
//! no token. Needs Apple Metal:
//!
//! ```text
//! cargo test --release --bin hf2q verify_decision
//! ```
//!
//! Run it also under `MTL_SHADER_VALIDATION=1` and
//! `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`; the printed digests must
//! agree.

use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::*;
use crate::inference::models::qwen35::decode_pool;
use crate::serve::sampler_pure::{sample_token, SamplingParams};

const WIDTH: usize = 32;

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

fn uniform(state: &mut u64) -> f32 {
    (next(state) & 0xff_ffff) as f32 / (1_u64 << 24) as f32
}

struct Case {
    vocab: usize,
    windows: Vec<JointWindow>,
    histories: Vec<Option<Vec<u32>>>,
    /// `[WIDTH, vocab]`.
    logits: Vec<f32>,
    /// Input token of every packed row.
    tokens: Vec<u32>,
}

impl Case {
    /// The sampler's `previous_tokens` for `row`, or `None` for a row of a
    /// window without a history.
    fn history(&self, row: usize) -> Option<Vec<u32>> {
        let (window, base) = self
            .windows
            .iter()
            .zip(&self.histories)
            .find(|(window, _)| (window.start..window.start + window.rows).contains(&row))?;
        let mut history = base.clone()?;
        history.extend_from_slice(&self.tokens[window.start + 1..=row]);
        let excess = history.len().saturating_sub(DECISION_SLOTS);
        Some(history.split_off(excess))
    }

    fn row(&self, row: usize) -> &[f32] {
        &self.logits[row * self.vocab..(row + 1) * self.vocab]
    }
}

fn greedy(penalty: f64) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: penalty,
        max_tokens: 1,
        seed: None,
    }
}

/// Run the decision dispatches over `case` and return each real row's
/// candidates and the tile's logits afterwards.
fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    case: &Case,
) -> (Vec<Option<JointRowDecision>>, Vec<f32>) {
    decode_pool::checked_reset_decode_pool().unwrap();
    let rows = case
        .windows
        .iter()
        .map(|window| window.start + window.rows)
        .max()
        .unwrap();
    let mut logits = device
        .alloc_buffer(WIDTH * case.vocab * 4, DType::F32, vec![WIDTH, case.vocab])
        .unwrap();
    logits
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&case.logits);
    let mut tokens = device
        .alloc_buffer(case.tokens.len() * 4, DType::U32, vec![case.tokens.len()])
        .unwrap();
    tokens
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&case.tokens);
    let histories = case
        .histories
        .iter()
        .map(|history| history.as_deref())
        .collect::<Vec<_>>();
    let mut enc = device.command_encoder().unwrap();
    let encoded = encode(
        &mut enc,
        registry,
        device,
        &logits,
        &case.windows,
        &histories,
        &tokens,
        rows,
        case.vocab,
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    let decisions = (0..rows)
        .map(|row| {
            let view: MlxBuffer = logits
                .slice_view((row * case.vocab * 4) as u64, case.vocab)
                .with_shape(vec![1, case.vocab])
                .unwrap();
            encoded.read(row, view).unwrap()
        })
        .collect();
    (decisions, logits.as_slice::<f32>().unwrap().to_vec())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Four windows covering 26 real rows of the 32-row tile; the third has no
/// history. Base histories of 0, 37 and 64 tokens, with duplicates.
fn layout(vocab: usize, state: &mut u64) -> Case {
    let windows = vec![
        JointWindow {
            slot: crate::serve::multi_seq_kv::SlotId(0),
            start: 0,
            rows: 8,
        },
        JointWindow {
            slot: crate::serve::multi_seq_kv::SlotId(1),
            start: 8,
            rows: 8,
        },
        JointWindow {
            slot: crate::serve::multi_seq_kv::SlotId(2),
            start: 16,
            rows: 3,
        },
        JointWindow {
            slot: crate::serve::multi_seq_kv::SlotId(3),
            start: 19,
            rows: 7,
        },
    ];
    let token = |state: &mut u64| (next(state) % vocab as u64) as u32;
    let base = |len: usize, state: &mut u64| -> Vec<u32> {
        let mut history: Vec<u32> = (0..len)
            .map(|_| (next(state) % vocab as u64) as u32)
            .collect();
        // Duplicates, as a repetition window has.
        for index in (3..len).step_by(5) {
            history[index] = history[index - 3];
        }
        history
    };
    let histories = vec![
        Some(base(64, state)),
        Some(base(37, state)),
        None,
        Some(base(0, state)),
    ];
    let tokens = (0..WIDTH).map(|_| token(state)).collect();
    let logits = (0..WIDTH * vocab)
        .map(|_| 24.0 * uniform(state) - 12.0)
        .collect();
    Case {
        vocab,
        windows,
        histories,
        logits,
        tokens,
    }
}

/// Give every decided row one of the structures a penalty turns into a
/// different decision.
fn make_adversarial(case: &mut Case, penalty: f64, state: &mut u64) {
    let vocab = case.vocab;
    let inverse = 1.0_f32 / penalty as f32;
    for row in 0..26 {
        let Some(history) = case.history(row) else {
            continue;
        };
        let logits = &mut case.logits[row * vocab..(row + 1) * vocab];
        let outside = |state: &mut u64, history: &[u32]| loop {
            let token = (next(state) % vocab as u64) as u32;
            if !history.contains(&token) {
                break token as usize;
            }
        };
        let pick = |state: &mut u64, history: &[u32]| {
            history[(next(state) % history.len() as u64) as usize] as usize
        };
        match (row % 8, history.is_empty()) {
            // A history token far on top: the penalty leaves it the winner.
            (0, false) => logits[pick(state, &history)] = 40.0,
            // A history token barely on top: the penalty hands the row to
            // the best other token.
            (1, false) => {
                let other = outside(state, &history);
                logits[other] = 20.0;
                logits[pick(state, &history)] = 20.5;
            }
            // A penalised history logit exactly equal to the best other
            // logit: the lower token id wins, whichever side it is on.
            (2 | 3, false) => {
                let token = pick(state, &history);
                logits[token] = 16.0;
                let tied = 16.0_f32 * inverse;
                let other = loop {
                    let other = outside(state, &history);
                    if (row % 8 == 2) == (other < token) {
                        break other;
                    }
                    if token == 0 || token == vocab - 1 {
                        break other;
                    }
                };
                logits[other] = tied;
            }
            // Two equal best logits outside the history.
            (4, _) => {
                let (first, second) = (outside(state, &history), outside(state, &history));
                logits[first] = 30.0;
                logits[second] = 30.0;
            }
            // Signed zeros on top of an all-negative row.
            (5, _) => {
                for value in logits.iter_mut() {
                    *value = -value.abs() - 1.0;
                }
                let (first, second) = (outside(state, &history), outside(state, &history));
                logits[first] = -0.0;
                logits[second] = 0.0;
                if let Some(&token) = history.first() {
                    logits[token as usize] = 0.0;
                }
            }
            // All negative, a history token on top: the penalty multiplies.
            (6, false) => {
                for value in logits.iter_mut() {
                    *value = -value.abs() - 1.0;
                }
                let other = outside(state, &history);
                logits[other] = -1.02;
                logits[pick(state, &history)] = -1.0;
            }
            _ => {}
        }
    }
}

fn digest(decisions: &[Option<JointRowDecision>], tile: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |word: u32| {
        hash = (hash ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3);
    };
    for decision in decisions.iter().flatten() {
        mix(decision.outside_token);
        mix(decision.outside_logit.to_bits());
        decision.history_tokens.iter().for_each(|&token| mix(token));
        decision
            .history_logits
            .iter()
            .for_each(|logit| mix(logit.to_bits()));
    }
    tile.iter().for_each(|value| mix(value.to_bits()));
    hash
}

/// Compare every row of `case` with the host sampler under `penalty`.
/// Returns the digest of everything the dispatches produced and how many
/// rows the penalty decided differently from a plain argmax.
fn check(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    case: &Case,
    penalty: f64,
) -> (u64, usize) {
    let (decisions, tile) = run(device, registry, case);
    let vocab = case.vocab;
    let mut flipped = 0;
    for (row, decision) in decisions.iter().enumerate() {
        let original = case.row(row);
        let after = &tile[row * vocab..(row + 1) * vocab];
        let Some(history) = case.history(row) else {
            assert!(
                decision.is_none(),
                "row {row} has candidates without a history"
            );
            assert!(
                bits(after) == bits(original),
                "row {row} changed without a history"
            );
            continue;
        };
        let decision = decision.as_ref().expect("a decided row has candidates");
        assert_eq!(
            decision.history_tokens, history,
            "row {row}: assembled history"
        );
        for token in 0..vocab {
            let expected = if history.contains(&(token as u32)) {
                MASKED.to_bits()
            } else {
                original[token].to_bits()
            };
            assert_eq!(after[token].to_bits(), expected, "row {row} token {token}");
        }
        let mut host_logits = original.to_vec();
        let expected = sample_token(&mut host_logits, &greedy(penalty), &history);
        assert_eq!(
            decision.greedy_token(&history, penalty),
            Some(expected),
            "row {row}, penalty {penalty}: the candidates decide another token"
        );
        let mut plain = original.to_vec();
        flipped += usize::from(sample_token(&mut plain, &greedy(1.0), &[]) != expected);
        assert!(
            bits(&decision.logits().unwrap()) == bits(original),
            "row {row}: fallback logits are not the head's"
        );
        // Any other history is refused, not approximated.
        let mut other = history.clone();
        other.push(0);
        assert_eq!(decision.greedy_token(&other, penalty), None);
    }
    // The untouched padding rows.
    assert!(bits(&tile[26 * vocab..]) == bits(&case.logits[26 * vocab..]));
    // The dispatches repeat their bits.
    let (again, tile_again) = run(device, registry, case);
    let digest = digest(&decisions, &tile);
    assert_eq!(
        digest,
        self::digest(&again, &tile_again),
        "not deterministic"
    );
    (digest, flipped)
}

#[test]
fn candidates_decide_what_the_host_sampler_decides() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut digests = Vec::new();
    for (seed, vocab) in [(1_u64, 4_099_usize), (2, 1_024), (3, 65), (4, 20_000)] {
        for penalty in [1.0, 1.05, 1.3, 0.8] {
            let mut state = seed * 7_919 + (penalty * 100.0) as u64;
            let mut case = layout(vocab, &mut state);
            digests.push(check(&device, &mut registry, &case, penalty).0);
            make_adversarial(&mut case, penalty, &mut state);
            let (digest, flipped) = check(&device, &mut registry, &case, penalty);
            digests.push(digest);
            if penalty > 1.0 {
                assert!(
                    flipped >= 3,
                    "vocab {vocab}, penalty {penalty}: the penalty changed only {flipped} decisions"
                );
            }
        }
    }
    let combined = digests
        .iter()
        .fold(0_u64, |hash, digest| hash.rotate_left(7) ^ digest);
    eprintln!("verify_decision digest small={combined:016x}");
}

/// The served vocabulary: 32 rows of 248,320 logits.
#[test]
fn candidates_decide_the_served_vocabulary() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut state = 97_u64;
    let mut case = layout(248_320, &mut state);
    make_adversarial(&mut case, 1.05, &mut state);
    let (digest, flipped) = check(&device, &mut registry, &case, 1.05);
    assert!(flipped >= 3);
    eprintln!("verify_decision digest served={digest:016x}");
}

#[test]
fn rows_the_candidates_cannot_decide_return_no_token() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut state = 11_u64;
    let mut case = layout(4_099, &mut state);
    let vocab = case.vocab;
    // (row, inside the history, value)
    let poisoned = [
        (1, false, f32::NAN),
        (2, true, f32::NAN),
        (3, false, f32::INFINITY),
        (4, true, f32::NEG_INFINITY),
        (9, false, f32::NEG_INFINITY),
        (10, true, f32::INFINITY),
    ];
    for (row, inside, value) in poisoned {
        let history = case.history(row).unwrap();
        let token = if inside {
            history[history.len() / 2] as usize
        } else {
            (0..vocab)
                .find(|token| !history.contains(&(*token as u32)))
                .unwrap()
        };
        case.logits[row * vocab + token] = value;
    }
    let (decisions, _) = run(&device, &mut registry, &case);
    for (row, decision) in decisions.iter().enumerate() {
        let Some(history) = case.history(row) else {
            continue;
        };
        let decision = decision.as_ref().unwrap();
        let refused = poisoned.iter().any(|&(poisoned, _, _)| poisoned == row);
        let token = decision.greedy_token(&history, 1.05);
        assert_eq!(token.is_none(), refused, "row {row}");
        if let Some(token) = token {
            let mut host = case.row(row).to_vec();
            assert_eq!(token, sample_token(&mut host, &greedy(1.05), &history));
        }
        // The fallback's logits are the head's, non-finite values included.
        assert!(
            bits(&decision.logits().unwrap()) == bits(case.row(row)),
            "row {row}"
        );
    }
}

/// The host half alone: candidates as the kernels would report them.
#[test]
fn the_host_finish_is_the_samplers_arithmetic() {
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let vocab = 16;
    let row = |values: &[f32]| {
        let mut buffer = device
            .alloc_buffer(vocab * 4, DType::F32, vec![1, vocab])
            .unwrap();
        buffer
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(values);
        buffer
    };
    // A penalty that overflows a history logit to negative infinity is the
    // sampler's own overflow.
    let mut logits = vec![-3.0_f32; vocab];
    logits[5] = -3.0e38;
    logits[9] = -2.0;
    let history = vec![5_u32, 9, 5];
    let mut masked = logits.clone();
    masked[5] = MASKED;
    masked[9] = MASKED;
    let decision = JointRowDecision::for_test(
        (0, -3.0),
        history.clone(),
        vec![-3.0e38, -2.0, -3.0e38],
        row(&masked),
    );
    for penalty in [1.0, 1.3, 2.0] {
        let mut host = logits.clone();
        assert_eq!(
            decision.greedy_token(&history, penalty),
            Some(sample_token(&mut host, &greedy(penalty), &history)),
            "penalty {penalty}"
        );
    }
    assert!(bits(&decision.logits().unwrap()) == bits(&logits));
    // A masked argmax that is itself the mask value decides nothing.
    let none = JointRowDecision::for_test((3, MASKED), vec![], vec![], row(&masked));
    assert_eq!(none.greedy_token(&[], 1.05), None);
    // The kernel's sentinel for a non-finite row.
    let invalid = JointRowDecision::for_test((u32::MAX, f32::NAN), vec![], vec![], row(&masked));
    assert_eq!(invalid.greedy_token(&[], 1.05), None);
}
