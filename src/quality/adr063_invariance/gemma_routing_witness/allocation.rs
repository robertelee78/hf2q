use super::*;
pub(super) fn allocate_bank(device: &MlxDevice) -> Result<Bank> {
    let make = |dtype, shape: Vec<usize>| {
        device.alloc_buffer(shape.iter().product::<usize>() * 4, dtype, shape)
    };
    let bank = Bank {
        scores: make(DType::F32, vec![ENTRY_CAP, 2])?,
        ids: make(DType::U32, vec![ENTRY_CAP, 2])?,
        status: make(DType::U32, vec![ENTRY_CAP])?,
        selected: make(DType::U32, vec![ENTRY_CAP, K])?,
        weights: make(DType::F32, vec![ENTRY_CAP, K])?,
    };
    let parent_bytes = [
        &bank.scores,
        &bank.ids,
        &bank.status,
        &bank.selected,
        &bank.weights,
    ]
    .iter()
    .try_fold(0usize, |n, b| n.checked_add(b.byte_len()))
    .context("parent byte overflow")?;
    ensure!(
        parent_bytes as u64 == PAYLOAD_BYTES,
        "actual routing bank parent capacities differ from pre-load budget"
    );
    Ok(bank)
}

pub fn install(
    device: &MlxDevice,
    root: &Path,
    identity: &Identity,
    acquisition: &Acquisition,
    mode: Mode,
    b: Option<&Budget>,
) -> Result<Option<Scope>> {
    validate_budget(b)?;
    if b.is_none() {
        return Ok(None);
    }
    ensure!(
        env!("HF2Q_MLX_NATIVE_VERSION") == WITNESS_NATIVE_VERSION,
        "exact published routing-witness native pin required; no path patch"
    );
    ensure!(
        root.is_absolute() && root.canonicalize()? == root,
        "actual private output root"
    );
    ensure!(
        acquisition.output_root == root
            && acquisition.actual_binary_sha256 == identity.binary_sha256
            && acquisition.device_registry_id == device.registry_id()
            && acquisition.required_model_opens == 3
            && acquisition.observed_model_opens >= 3
            && acquisition.provenance == "IndependentRootReviewedComposition",
        "same actual reviewed acquisition/device bank owner"
    );
    ACTIVE.with(|s| {
        ensure!(s.borrow().is_none(), "nested routing bank");
        Ok::<_, anyhow::Error>(())
    })?;
    let bank = allocate_bank(device)?;
    let meta = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join("routing-witness.jsonl"))?;
    let mut a = Active {
        bank,
        records: Vec::with_capacity(ENTRY_CAP),
        call: None,
        seen: BTreeSet::new(),
        mode,
        next: 0,
        read: 0,
        pending: false,
        meta,
        metadata: 0,
    };
    let parents=[("scores",&a.bank.scores),("boundary_ids",&a.bank.ids),("status",&a.bank.status),
        ("selected_ids",&a.bank.selected),("selected_weights",&a.bank.weights)].into_iter().map(|(role,b)|
        json!({"role":role,"parent_cpu_base":b.contents_ptr() as usize as u64,"parent_bytes":b.byte_len(),
            "logical_bytes":b.data_byte_len(),"offset":b.byte_offset(),"shape":b.shape(),"dtype":format!("{:?}",b.dtype())})).collect::<Vec<_>>();
    event(
        &mut a,
        json!({"kind":"header","identity":identity,"acquisition":acquisition,"mode":mode,
        "payload_limit":PAYLOAD_BYTES,"metadata_limit":METADATA_BYTES,"cpu_ledger_allowance":CPU_LEDGER_BYTES,"bank_parents":parents,
        "selection_domain":"actual softmax before topK renorm/per-expert scale","native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "noncanonical_input_join":"original actual cancellation/transaction journal; never assume canonical after fork",
        "physical_membership_claim":false,"extra_waits":0,"qualified":false}),
    )?;
    ACTIVE.with(|s| *s.borrow_mut() = Some(a));
    Ok(Some(Scope {
        _same_thread: std::marker::PhantomData,
    }))
}
