//! Explicit primitive authority ONLY. No model Identity/acquisition fabrication.
use super::*;
const GUARD: u32 = 0x5a39_a51c;
const OVERWRITE: u32 = 0x4f12_3abc;
fn u32_buffer(device: &MlxDevice, elements: usize, value: u32) -> Result<MlxBuffer> {
    let mut b = device.alloc_buffer(elements * 4, DType::U32, vec![elements])?;
    b.as_mut_slice::<u32>()?.fill(value);
    Ok(b)
}
fn f32_buffer(device: &MlxDevice, values: &[f32]) -> Result<MlxBuffer> {
    let mut b = device.alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])?;
    b.as_mut_slice::<f32>()?.copy_from_slice(values);
    Ok(b)
}
fn guard_ok(b: &MlxBuffer) -> Result<()> {
    let x = b.as_logical_slice::<u32>()?;
    ensure!(
        x[..4].iter().chain(&x[x.len() - 4..]).all(|v| *v == GUARD),
        "nonzero offset output guard"
    );
    Ok(())
}
fn primitive_scope(device: &MlxDevice, meta: File) -> Result<Scope> {
    ACTIVE.with(|s| {
        ensure!(s.borrow().is_none(), "no active model/primitive bank");
        Ok::<_, anyhow::Error>(())
    })?;
    // Exact SAME actual production diagnostic allocator, views, producer and
    // stable-copy code. Only this private fixture omits model authority.
    let mut a = Active {
        bank: allocation::allocate_bank(device)?,
        records: Vec::with_capacity(LAYERS * 4),
        call: None,
        seen: BTreeSet::new(),
        mode: Mode::Ordinary,
        next: 0,
        read: 0,
        pending: false,
        meta,
        metadata: 0,
    };
    a.bank.scores.as_mut_slice::<f32>()?.fill(f32::NAN);
    a.bank.ids.as_mut_slice::<u32>()?.fill(u32::MAX);
    a.bank.status.as_mut_slice::<u32>()?.fill(u32::MAX);
    a.bank.selected.as_mut_slice::<u32>()?.fill(u32::MAX);
    a.bank.weights.as_mut_slice::<f32>()?.fill(f32::NAN);
    event(
        &mut a,
        json!({"kind":"primitive_only_header","model_identity":null,"acquisition":null,
        "model_loaded":false,"full_matrix":false,"native":env!("HF2Q_MLX_NATIVE_VERSION"),
        "bank_payload":PAYLOAD_BYTES,"GPU_waits_per_case":1}),
    )?;
    ACTIVE.with(|s| *s.borrow_mut() = Some(a));
    Ok(Scope {
        _same_thread: std::marker::PhantomData,
    })
}
#[test]
#[ignore = "Metal scalar/batch HF bank copy + GPU scratch overwrite, primitive only"]
fn actual_scalar_and_batch_bank_copy_survives_gpu_scratch_overwrite() -> Result<()> {
    // Fixed user-requested Metal parent payload; no model, no hidden geometry.
    // Native parameter/encoder/driver overhead is NOT a physical-fit claim.
    const NAMED_UPPER: u64 = PAYLOAD_BYTES
        + (4 * E * 4 + E * 4 + 2 * (4 * K + 8) * 4 + 2 * (LAYERS * 4 * K + 8) * 4 + 2 * 4 * K * 4)
            as u64;
    ensure!(
        NAMED_UPPER == 2_792_960 && NAMED_UPPER < 4 * 1024 * 1024,
        "pre-device fixed parent budget"
    );
    ensure!(
        env!("HF2Q_MLX_NATIVE_VERSION") == WITNESS_NATIVE_VERSION,
        "published exact native gate"
    );
    // No retirement proof between cases: conservatively retain BOTH named
    // case payloads in the whole-test requested-root budget.
    const BOTH_CASES_UPPER:u64=2*NAMED_UPPER;
    ensure!(BOTH_CASES_UPPER==5_585_920&&BOTH_CASES_UPPER<8*1024*1024,"whole-test no-retirement named parent sum");
    let device = MlxDevice::new()?;
    let mut registry = KernelRegistry::new();
    let tmp = tempfile::Builder::new()
        .prefix("hf2q-q1-routing-primitive-")
        .tempdir()?
        .keep();
    println!("Q1_ROUTING_BANK_PRIMITIVE_BEGIN metadata_root={} per_case_named_parent_upper={NAMED_UPPER} both_cases_no_retirement_upper={BOTH_CASES_UPPER} model=false",tmp.display());
    for n in [1usize, 4] {
        let metadata = tmp.join(format!("primitive-{n}.jsonl"));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&metadata)?;
        let scope = primitive_scope(&device, file)?;
        let rows = (0..n)
            .map(|lane| Row {
                lane,
                position: 13 + lane,
            })
            .collect::<Vec<_>>();
        begin(Namespace::Canonical, rows.clone())?;
        let values = (0..n * E)
            .map(|i| {
                let row = i / E;
                let e = i % E;
                (((e * 37 + row * 23) % E) as f32) * 0.017 + row as f32 * 0.013
            })
            .collect::<Vec<_>>();
        let logits = f32_buffer(&device, &values)?.with_shape(vec![n, E])?;
        let scales = f32_buffer(
            &device,
            &(0..E).map(|e| 0.5 + e as f32 / 256.0).collect::<Vec<_>>(),
        )?;
        let scratch_ids = u32_buffer(&device, n * K + 8, GUARD)?;
        // Weights have F32 dtype but identical raw guarded allocation.
        let mut scratch_weights_f32 =
            device.alloc_buffer((n * K + 8) * 4, DType::F32, vec![n * K + 8])?;
        scratch_weights_f32.as_mut_slice::<u32>()?.fill(GUARD);
        let old_ids = u32_buffer(&device, LAYERS * n * K + 8, GUARD)?;
        let mut old_weights = device.alloc_buffer(
            (LAYERS * n * K + 8) * 4,
            DType::F32,
            vec![LAYERS * n * K + 8],
        )?;
        old_weights.as_mut_slice::<u32>()?.fill(GUARD);
        let ids = scratch_ids.slice_view(16, n * K).with_shape(vec![n, K])?;
        let weights = scratch_weights_f32
            .slice_view(16, n * K)
            .with_shape(vec![n, K])?;
        let poison_ids = u32_buffer(&device, n * K, OVERWRITE)?;
        let mut poison_weights = device.alloc_buffer(n * K * 4, DType::F32, vec![n, K])?;
        poison_weights.as_mut_slice::<u32>()?.fill(OVERWRITE);
        let mut enc = device.command_encoder()?;
        let mut variant = None;
        for layer in 0..LAYERS {
            let old_i = old_ids
                .slice_view((16 + layer * n * K * 4) as u64, n * K)
                .with_shape(vec![n, K])?;
            let old_w = old_weights
                .slice_view((16 + layer * n * K * 4) as u64, n * K)
                .with_shape(vec![n, K])?;
            enc.memory_barrier();
            let baseline = if n == 1 {
                dispatch_gemma_with_witness(
                    &mut enc,
                    &mut registry,
                    &device,
                    &logits,
                    &old_i,
                    &old_w,
                    &scales,
                    E as u32,
                    K as u32,
                    None,
                )?
            } else {
                dispatch_gemma_batch_with_witness(
                    &mut enc,
                    &mut registry,
                    &device,
                    &logits,
                    &old_i,
                    &old_w,
                    &scales,
                    E as u32,
                    K as u32,
                    n as u32,
                    None,
                )?
            };
            ensure!(!baseline.witness_encoded, "old None must stay inactive");
            variant = Some(format!("{:?}", baseline.variant));
            enc.memory_barrier();
            let routed = if n == 1 {
                scalar(
                    &mut enc,
                    &mut registry,
                    &device,
                    &logits,
                    &ids,
                    &weights,
                    &scales,
                    E as u32,
                    K as u32,
                    layer,
                    rows[0].position,
                    None,
                    0,
                )?
            } else {
                let layout = rows
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (r.lane as u32, r.position, 1, i))
                    .collect::<Vec<_>>();
                batch(
                    &mut enc,
                    &mut registry,
                    &device,
                    &logits,
                    &ids,
                    &weights,
                    &scales,
                    E as u32,
                    K as u32,
                    layer,
                    0,
                    n,
                    Some(layout),
                )?
            };
            ensure!(routed, "actual HF wrapper must use active same producer");
            // GPU overwrite AFTER the production wrapper's same-encoder copy,
            // before any wait/CPU read. No retained clone of scratch as oracle.
            enc.blit_copy_bytes(&poison_ids, 0, &ids, 0, (n * K * 4) as u64)?;
            enc.blit_copy_bytes(&poison_weights, 0, &weights, 0, (n * K * 4) as u64)?;
        }
        enc.commit_and_wait_labeled("q1.routing_primitive.single_checked_drain")?;
        drop(enc);
        output::primitive_end_checked()?;
        guard_ok(&scratch_ids)?;
        guard_ok(&scratch_weights_f32)?;
        guard_ok(&old_ids)?;
        guard_ok(&old_weights)?;
        ensure!(
            ids.as_logical_slice::<u32>()?
                .iter()
                .chain(weights.as_logical_slice::<u32>()?)
                .all(|v| *v == OVERWRITE),
            "actual GPU overwrite occurred"
        );
        ACTIVE.with(|s| -> Result<()> {
            let s = s.borrow();
            let a = s.as_ref().context("actual bank absent")?;
            ensure!(
                !a.pending
                    && a.call.is_none()
                    && a.read == LAYERS * n
                    && a.records.len() == LAYERS * n,
                "all30 actual layer rows checked"
            );
            let reference_i = old_ids.as_logical_slice::<u32>()?;
            let reference_w = old_weights.as_logical_slice::<u32>()?;
            for entry in 0..LAYERS * n {
                let (_, _, _, selected, weights) = views(&a.bank, entry, 1)?;
                ensure!(
                    selected.as_logical_slice::<u32>()?
                        == &reference_i[4 + entry * K..4 + (entry + 1) * K],
                    "same selected ID/order"
                );
                ensure!(
                    weights.as_logical_slice::<u32>()?
                        == &reference_w[4 + entry * K..4 + (entry + 1) * K],
                    "same selected weight bits"
                );
                ensure!(
                    Some(a.records[entry].native_variant.to_string()) == variant,
                    "actual Some/None variant"
                );
            }
            ensure!(
                a.bank.selected.as_logical_slice::<u32>()?[LAYERS * n * K..]
                    .iter()
                    .all(|x| *x == u32::MAX),
                "untouched bank tail guard"
            );
            for (buffer, width, sentinel) in [
                (&a.bank.scores, 2, f32::NAN.to_bits()),
                (&a.bank.ids, 2, u32::MAX),
                (&a.bank.status, 1, u32::MAX),
                (&a.bank.weights, K, f32::NAN.to_bits()),
            ] {
                ensure!(
                    buffer.as_logical_slice::<u32>()?[LAYERS * n * width..]
                        .iter()
                        .all(|x| *x == sentinel),
                    "all five untouched bank tails"
                );
            }
            Ok(())
        })?;
        ensure!(
            output::finish().is_err(),
            "primitive rows cannot qualify full matrix"
        );
        begin(
            Namespace::Canonical,
            vec![Row {
                lane: 0,
                position: 31,
            }],
        )?;
        ensure!(
            output::primitive_end_checked().is_err(),
            "missing30layer coverage must refuse before CPU read"
        );
        drop(scope);
        ensure!(
            std::fs::metadata(&metadata)?.len() < 1024 * 1024,
            "bounded primitive metadata"
        );
        println!("Q1_ROUTING_BANK_PRIMITIVE rows={n} layers=30 variant={} exact_ids_weights=true scratch_gpu_overwritten=true checked_drains=1 missing_coverage_refused=true full_matrix=false per_case_named_parent_upper={NAMED_UPPER} both_cases_no_retirement_upper={BOTH_CASES_UPPER}",variant.unwrap());
    }
    Ok(())
}
