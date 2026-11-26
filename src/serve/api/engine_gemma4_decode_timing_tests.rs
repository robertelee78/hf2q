//! Q4-GEMMA-DECODE M1. Native GPU-busy intervals, not host wall-clock.
//! The main probe runs 3 x 512 measured steps per width/arm after 32 warmups.
//! Alternate first arm by round; router capture stays off. The extra process
//! sets HF2Q_FUSE_LMHEAD=0 for one separate-head round, disclosed not re-gated.
use super::*;

const WARMUP: usize = 32;
const MEASURED: usize = 512;

fn median(values: &mut [u64]) -> f64 {
    values.sort_unstable();
    let n = values.len();
    if n % 2 == 0 { (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0 }
    else { values[n / 2] as f64 }
}

fn run_probe(rounds: usize, separate_head: bool) {
    assert_eq!(std::env::var("HF2Q_GPU_BUSY").as_deref(), Ok("1"),
        "real Native GPU intervals must be enabled before model acquisition");
    assert_eq!(std::env::var("HF2Q_FUSE_LMHEAD").as_deref() == Ok("0"), separate_head);
    let output = PathBuf::from(std::env::var("HF2Q_GEMMA_DECODE_PROBE_OUTPUT")
        .expect("unique supervised output path required"));
    assert!(!output.exists(), "refuse overwriting prior M1 evidence");
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut loaded = load_model();
    let LoadedModel::Gemma(model) = &mut loaded else { unreachable!() };
    let mut stock = provision(model, false);
    let mut candidate = provision(model, true);
    let slots = [SlotId(0), SlotId(1), SlotId(2), SlotId(3)];
    let mut records = Vec::with_capacity(rounds * 2);
    let mut width_summaries = Vec::with_capacity(2);
    for width in [4, 1] {
        let mut stock_times = Vec::with_capacity(rounds * MEASURED);
        let mut candidate_times = Vec::with_capacity(rounds * MEASURED);
        for round in 0..rounds {
            let mut tokens = [0u32; SLOTS];
            let mut positions = [512usize; SLOTS];
            for slot in 0..width {
                let input = prompt(512, round * SLOTS + slot + 16);
                let a = prefill(model, &mut stock, slots[slot], &input, WARMUP + MEASURED);
                let b = prefill(model, &mut candidate, slots[slot], &input, WARMUP + MEASURED);
                assert_eq!(a, b, "M1 identical prefill first token");
                assert_cache_equal(model, &stock, &candidate, slot, 0..input.len());
                tokens[slot] = a;
            }
            let mut samples = Vec::with_capacity(MEASURED);
            let mut fused_head_seen = None;
            for step in 0..WARMUP + MEASURED {
                let (a, b) = if round % 2 == 0 {
                    let a = decode_step(model, &mut stock, &slots[..width],
                        &tokens[..width], &positions[..width], false);
                    let b = decode_step(model, &mut candidate, &slots[..width],
                        &tokens[..width], &positions[..width], false);
                    (a, b)
                } else {
                    let b = decode_step(model, &mut candidate, &slots[..width],
                        &tokens[..width], &positions[..width], false);
                    let a = decode_step(model, &mut stock, &slots[..width],
                        &tokens[..width], &positions[..width], false);
                    (a, b)
                };
                assert_logits_equal(&a, &b);
                assert_eq!(a.fused_head, b.fused_head);
                assert!(a.body_gpu_ns > 0 && b.body_gpu_ns > 0,
                    "missing actual healthy GPU intervals is not a zero-time measurement");
                assert!(a.router.is_none() && b.router.is_none());
                if separate_head {
                    assert!(!a.fused_head && a.head_gpu_ns > 0 && b.head_gpu_ns > 0);
                }
                match fused_head_seen {
                    None => fused_head_seen = Some(a.fused_head),
                    Some(value) => assert_eq!(value, a.fused_head),
                }
                if step >= WARMUP {
                    samples.push([a.body_gpu_ns, a.head_gpu_ns, b.body_gpu_ns, b.head_gpu_ns]);
                    stock_times.push(a.body_gpu_ns + a.head_gpu_ns);
                    candidate_times.push(b.body_gpu_ns + b.head_gpu_ns);
                }
                let vocab = a.logits.len() / width;
                for slot in 0..width {
                    tokens[slot] = argmax(&a.logits[slot * vocab..(slot + 1) * vocab]);
                    positions[slot] += 1;
                }
            }
            assert_eq!(samples.len(), MEASURED);
            records.push(serde_json::json!({
                "width":width,"round":round,"first_arm":if round%2==0 {"stock"}else{"candidate"},
                "warmup_steps_discarded":WARMUP,"measured_steps_per_arm":MEASURED,
                "body_interval_includes_fused_head":fused_head_seen.unwrap(),
                "gpu_ns_columns":["stock_body","stock_separate_head","candidate_body","candidate_separate_head"],
                "gpu_ns":samples,
            }));
        }
        let stock_median = median(&mut stock_times);
        let candidate_median = median(&mut candidate_times);
        width_summaries.push(serde_json::json!({"width":width,
            "samples_per_arm":stock_times.len(),"stock_median_step_gpu_ns":stock_median,
            "candidate_median_step_gpu_ns":candidate_median,
            "candidate_stock_ratio":candidate_median/stock_median}));
    }
    let width4 = width_summaries[0]["candidate_stock_ratio"].as_f64().unwrap();
    let width1 = width_summaries[1]["candidate_stock_ratio"].as_f64().unwrap();
    let passed = width4 <= 0.90 && width1 <= 1.01;
    let report = serde_json::json!({
        "schema":"hf2q.q4.gemma-decode-in-situ.v1","context_per_slot":2048,
        "model_weights_loaded_once":true,"independent_stock_candidate_KV":true,
        "changed_policy":"only frozen decode_grouped_geglu selector, stock=false/candidate=true",
        "same_source_binary_native_model_inputs":true,"rounds_per_width":rounds,
        "separate_head_disclosure_run":separate_head,"router_capture":false,
        "metric":"candidate/stock ratio of pooled per-arm median actual decode-step GPU-busy ns",
        "actual_logits_u32_equal_every_step":true,"records":records,"widths":width_summaries,
        "gated_run":!separate_head,"m1_thresholds":{"width4_max_ratio":0.90,"width1_max_ratio":1.01},
        "m1_threshold_pass":if separate_head {None}else{Some(passed)},
        "whole_story_qualified":false,"production_Q5_qualified":false,
    });
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&output)
        .expect("create unique bounded M1 evidence");
    file.write_all(&serde_json::to_vec_pretty(&report).unwrap()).expect("publish actual M1 evidence");
    eprintln!("GEMMA_DECODE_M1 rounds={rounds} steps={MEASURED} separate_head={separate_head} width4_ratio={width4:.9} width1_ratio={width1:.9} output={}",output.display());
    if !separate_head {
        assert!(passed, "M1 failed: width4 {width4:.9} >0.90 or width1 {width1:.9} >1.01; stop, no retry");
    }
}

#[test]
#[ignore = "requires approved Gemma GGUF, explicit F2, HF2Q_GPU_BUSY=1 and the exclusive GPU lane"]
fn gemma_decode_kernels_in_situ_probe() { run_probe(3, false); }

#[test]
#[ignore = "extra M1 disclosure: requires HF2Q_FUSE_LMHEAD=0 and a unique supervised output"]
fn gemma_decode_kernels_separate_head_probe() { run_probe(1, true); }
