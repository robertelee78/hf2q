use super::*;

#[path = "engine_gemma4_tiny_prefill_test_support.rs"]
mod support;
use support::*;

/// Direct decode -> tiny-cold-prefill kernel discriminator for the
/// intermittent N=8 worker fault. Without constructing scheduler/reply
/// ownership, it mirrors both relevant GPU transaction shapes: eight tiny
/// cold prefills followed by a width-eight batched decode, and staggered
/// decode widths one through eight around later tiny admissions. Every cold
/// prefill and the coalesced width-eight decode must match independent N=1
/// references; staircase decode rows must remain finite before the following
/// cold prefill checks for persistent corruption.
#[test]
fn gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant() {
    if skip_unless_gated("gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant") {
        return;
    }
    let gguf_path: PathBuf = std::env::var(BYTE_EQUIV_E2E_GGUF_ENV)
        .map(PathBuf::from)
        .expect("HF2Q_BYTE_EQUIV_E2E_GGUF set");
    assert!(gguf_path.exists(), "GGUF missing: {}", gguf_path.display());
    let load_opts = LoadOptions {
        model_path: gguf_path,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
            glp_path: None,
            glp_alpha: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
    };
    let prompts: Vec<Vec<u32>> = vec![
        vec![1, 2, 3],
        vec![4, 5, 6, 7],
        vec![8, 9],
        vec![10, 11, 12, 13, 14],
        vec![15, 16],
        vec![17, 18, 19, 20],
        vec![21, 22, 23],
        vec![24, 25, 26, 27, 28],
    ];
    let max_decode_tokens = 24usize;

    let expected: Vec<(u32, u32)> = prompts
        .iter()
        .map(|prompt| {
            let mut loaded = LoadedModel::load(&load_opts).expect("load N=1 reference");
            let LoadedModel::Gemma(g) = &mut loaded else {
                panic!("expected Gemma GGUF")
            };
            g.provision_multi_seq_kv_for_slot_aware(1)
                .expect("provision N=1 reference KV");
            let mut kv = g.multi_seq_kv.take().expect("N=1 HB scaffold");
            let mut hybrid = g.multi_seq_kv_hybrid.take();
            let mut dense = g.multi_seq_kv_dense.take();
            let mut mlx = g.multi_seq_kv_mlx.take();
            let first = g
                .weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    SlotId(0),
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .expect("N=1 reference prefill");
            clear_gemma4_self_mounts(g);
            let mut profile = None;
            let second = g
                .weights
                .forward_decode_slot_aware(
                    first,
                    prompt.len(),
                    &mut g.ctx,
                    &mut profile,
                    SlotId(0),
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .expect("N=1 reference decode");
            (first, second)
        })
        .collect();
    assert!(
        expected
            .iter()
            .all(|&(first, second)| first != 0 && second != 0),
        "vacuous: an N=1 reference selected token 0: {expected:?}"
    );
    let resume_base: Vec<u32> = (100u32..140).collect();
    let resume_prompts: Vec<Vec<u32>> = [vec![41u32], vec![42u32, 43], vec![44u32, 45, 46, 47, 48]]
        .into_iter()
        .map(|suffix| {
            let mut prompt = resume_base.clone();
            prompt.extend(suffix);
            prompt
        })
        .collect();
    let expected_resumes: Vec<u32> = resume_prompts
        .iter()
        .map(|prompt| {
            let mut loaded = LoadedModel::load(&load_opts).expect("load resume reference");
            let LoadedModel::Gemma(g) = &mut loaded else {
                panic!("expected Gemma GGUF")
            };
            g.provision_multi_seq_kv_for_slot_aware(1)
                .expect("provision resume reference KV");
            let mut kv = g.multi_seq_kv.take().expect("resume reference HB scaffold");
            let mut hybrid = g.multi_seq_kv_hybrid.take();
            let mut dense = g.multi_seq_kv_dense.take();
            let mut mlx = g.multi_seq_kv_mlx.take();
            g.weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    SlotId(0),
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .expect("cold full-prompt resume reference")
        })
        .collect();
    assert!(
        expected_resumes.iter().all(|&token| token != 0),
        "vacuous: a cold resume reference selected token 0: {expected_resumes:?}"
    );
    let boundary_prompts: Vec<Vec<u32>> = [31usize, 32usize]
        .into_iter()
        .map(|len| (1..=len as u32).collect())
        .collect();
    let expected_boundaries: Vec<u32> = boundary_prompts
        .iter()
        .map(|prompt| {
            let mut loaded = LoadedModel::load(&load_opts).expect("load boundary reference");
            let LoadedModel::Gemma(g) = &mut loaded else {
                panic!("expected Gemma GGUF")
            };
            g.provision_multi_seq_kv_for_slot_aware(1)
                .expect("provision boundary reference KV");
            let mut kv = g
                .multi_seq_kv
                .take()
                .expect("boundary reference HB scaffold");
            let mut hybrid = g.multi_seq_kv_hybrid.take();
            let mut dense = g.multi_seq_kv_dense.take();
            let mut mlx = g.multi_seq_kv_mlx.take();
            g.weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    SlotId(0),
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .expect("N=1 boundary reference")
        })
        .collect();

    let mut loaded = LoadedModel::load(&load_opts).expect("load N=8 discriminator");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("expected Gemma GGUF")
    };
    g.provision_multi_seq_kv_for_slot_aware(8)
        .expect("provision N=8 KV");
    let mut kv = g.multi_seq_kv.take().expect("N=8 HB scaffold");
    let mut hybrid = g.multi_seq_kv_hybrid.take();
    let mut dense = g.multi_seq_kv_dense.take();
    let mut mlx = g.multi_seq_kv_mlx.take();
    let repeats = positive_test_env_usize("HF2Q_GEMMA_N8_PREFILL_REPEATS", 256, 4_096);
    assert!(
        dense.is_none(),
        "tiny-prefill discriminator requires HF2Q_USE_DENSE=0"
    );
    assert!(
        mlx.is_none(),
        "tiny-prefill discriminator requires HF2Q_TQ_CODEBOOK_BITS=8"
    );
    let actual_regime = if hybrid.is_some() {
        "hybrid"
    } else {
        "full-tq"
    };
    let expected_regime = match std::env::var("HF2Q_GEMMA_N8_EXPECTED_KV_REGIME").as_deref() {
        Ok("hybrid") => "hybrid",
        Ok("full-tq") => "full-tq",
        Ok(other) => {
            panic!("HF2Q_GEMMA_N8_EXPECTED_KV_REGIME must be hybrid or full-tq, got {other:?}")
        }
        Err(error) => panic!("HF2Q_GEMMA_N8_EXPECTED_KV_REGIME is required: {error}"),
    };
    assert_eq!(
        actual_regime, expected_regime,
        "tiny-prefill discriminator did not provision the requested KV regime"
    );
    eprintln!("[gemma-n8-tiny-prefill] kv_regime={actual_regime} cold_rounds={repeats}");

    for iteration in 0..repeats {
        // Canonical cross-slot admission coalesces the eight cold requests,
        // then the tiny-work containment runs their linear prefills before
        // the scheduler's first width-eight batched decode.
        for slot_idx in 0..8 {
            let slot_id = SlotId(slot_idx as u32);
            for buffer in &mut kv {
                buffer.reset_for_slot(slot_id).expect("reset N=8 HB slot");
            }
            if let Some(hybrid) = hybrid.as_mut() {
                for buffer in hybrid {
                    buffer
                        .reset_for_slot(slot_id)
                        .expect("reset N=8 hybrid slot");
                }
            }
        }

        let mut feed_tokens = Vec::with_capacity(8);
        let mut positions = Vec::with_capacity(8);
        for (slot_idx, prompt) in prompts.iter().enumerate() {
            let slot_id = SlotId(slot_idx as u32);
            clear_gemma4_self_mounts(g);
            let actual = g
                .weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    slot_id,
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "coalesced prefill failed at iteration {iteration}, slot {slot_idx}: {error:#}"
                    )
                });
            clear_gemma4_self_mounts(g);
            assert_eq!(
                actual, expected[slot_idx].0,
                "coalesced prefill diverged at iteration {iteration}, slot {slot_idx}"
            );
            feed_tokens.push(actual);
            positions.push(prompt.len());
        }
        gemma4_test_decode_rows(
            g,
            kv.as_mut_slice(),
            hybrid.as_mut().map(Vec::as_mut_slice),
            feed_tokens.as_mut_slice(),
            positions.as_mut_slice(),
            8,
            &format!("coalesced iteration {iteration}, width 8"),
        );
        assert_eq!(
            feed_tokens,
            expected
                .iter()
                .map(|&(_, second)| second)
                .collect::<Vec<_>>(),
            "coalesced width-eight decode diverged at iteration {iteration}"
        );

        // Non-coalesced admission alternates one new cold lane with the
        // scheduler's current active decode width. Cover every width 1..=8;
        // a following prefill catches persistent corruption from the prior
        // transaction even when its logits remained finite.
        for slot_idx in 0..8 {
            let slot_id = SlotId(slot_idx as u32);
            for buffer in &mut kv {
                buffer.reset_for_slot(slot_id).expect("reset N=8 HB slot");
            }
            if let Some(hybrid) = hybrid.as_mut() {
                for buffer in hybrid {
                    buffer
                        .reset_for_slot(slot_id)
                        .expect("reset N=8 hybrid slot");
                }
            }
        }
        feed_tokens.clear();
        positions.clear();
        for (slot_idx, prompt) in prompts.iter().enumerate() {
            if slot_idx >= 1 {
                gemma4_test_decode_rows(
                    g,
                    kv.as_mut_slice(),
                    hybrid.as_mut().map(Vec::as_mut_slice),
                    feed_tokens.as_mut_slice(),
                    positions.as_mut_slice(),
                    slot_idx,
                    &format!("staircase iteration {iteration}, width {slot_idx}"),
                );
            }
            let slot_id = SlotId(slot_idx as u32);
            clear_gemma4_self_mounts(g);
            let actual = g
                .weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    slot_id,
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "staircase prefill failed at iteration {iteration}, slot {slot_idx}: {error:#}"
                    )
                });
            clear_gemma4_self_mounts(g);
            assert_eq!(
                actual, expected[slot_idx].0,
                "staircase prefill diverged at iteration {iteration}, slot {slot_idx}"
            );
            feed_tokens.push(actual);
            positions.push(prompt.len());
        }
        gemma4_test_decode_rows(
            g,
            kv.as_mut_slice(),
            hybrid.as_mut().map(Vec::as_mut_slice),
            feed_tokens.as_mut_slice(),
            positions.as_mut_slice(),
            8,
            &format!("staircase iteration {iteration}, width 8"),
        );
    }

    let resume_repeats = positive_test_env_usize("HF2Q_GEMMA_N8_RESUME_REPEATS", 64, 1_024);
    eprintln!("[gemma-n8-tiny-prefill] retained_suffix_rounds={resume_repeats}");
    for iteration in 0..resume_repeats {
        let mut feed_tokens = Vec::with_capacity(2);
        let mut positions = Vec::with_capacity(2);
        for (slot_idx, prompt) in prompts[..2].iter().enumerate() {
            let slot_id = SlotId(slot_idx as u32);
            for buffer in &mut kv {
                buffer
                    .reset_for_slot(slot_id)
                    .expect("reset active HB slot before resume discriminator");
            }
            if let Some(hybrid) = hybrid.as_mut() {
                for buffer in hybrid {
                    buffer
                        .reset_for_slot(slot_id)
                        .expect("reset active hybrid slot before resume discriminator");
                }
            }
            clear_gemma4_self_mounts(g);
            let first = g
                .weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    slot_id,
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .expect("prefill active row before resume discriminator");
            clear_gemma4_self_mounts(g);
            feed_tokens.push(first);
            positions.push(prompt.len());
        }

        for (case, (prompt, &expected)) in resume_prompts
            .iter()
            .zip(expected_resumes.iter())
            .enumerate()
        {
            let resume_slot = SlotId(2);
            for buffer in &mut kv {
                buffer
                    .reset_for_slot(resume_slot)
                    .expect("reset HB resume slot");
            }
            if let Some(hybrid) = hybrid.as_mut() {
                for buffer in hybrid {
                    buffer
                        .reset_for_slot(resume_slot)
                        .expect("reset hybrid resume slot");
                }
            }
            clear_gemma4_self_mounts(g);
            g.weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    &resume_base,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    resume_slot,
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .expect("prefill exact live prefix");
            clear_gemma4_self_mounts(g);

            gemma4_test_decode_rows(
                g,
                kv.as_mut_slice(),
                hybrid.as_mut().map(Vec::as_mut_slice),
                feed_tokens.as_mut_slice(),
                positions.as_mut_slice(),
                2,
                &format!("resume iteration {iteration}, suffix case {case}"),
            );
            clear_gemma4_self_mounts(g);
            let actual = g
                .weights
                .forward_prefill_with_soft_tokens_slot_aware_resume(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    resume_slot,
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                    resume_base.len(),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "tiny live resume failed at iteration {iteration}, suffix case {case}: {error:#}"
                    )
                });
            clear_gemma4_self_mounts(g);
            assert_eq!(
                actual, expected,
                "tiny live resume diverged at iteration {iteration}, suffix case {case}"
            );
        }

        for (case, (prompt, &expected)) in boundary_prompts
            .iter()
            .zip(expected_boundaries.iter())
            .enumerate()
        {
            let boundary_slot = SlotId(2);
            for buffer in &mut kv {
                buffer
                    .reset_for_slot(boundary_slot)
                    .expect("reset HB boundary slot");
            }
            if let Some(hybrid) = hybrid.as_mut() {
                for buffer in hybrid {
                    buffer
                        .reset_for_slot(boundary_slot)
                        .expect("reset hybrid boundary slot");
                }
            }
            gemma4_test_decode_rows(
                g,
                kv.as_mut_slice(),
                hybrid.as_mut().map(Vec::as_mut_slice),
                feed_tokens.as_mut_slice(),
                positions.as_mut_slice(),
                2,
                &format!("boundary iteration {iteration}, case {case}"),
            );
            clear_gemma4_self_mounts(g);
            let actual = g
                .weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    max_decode_tokens,
                    &mut g.ctx,
                    boundary_slot,
                    &mut kv,
                    hybrid.as_mut(),
                    dense.as_mut(),
                    mlx.as_mut(),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "boundary prefill failed at iteration {iteration}, case {case}: {error:#}"
                    )
                });
            clear_gemma4_self_mounts(g);
            assert_eq!(
                actual, expected,
                "boundary prefill diverged at iteration {iteration}, case {case}"
            );
        }
    }
}
