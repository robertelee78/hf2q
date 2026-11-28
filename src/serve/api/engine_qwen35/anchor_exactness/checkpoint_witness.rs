//! Original checkpoint rows are registered before their native prefill. Equality
//! consumes these exact retained rows; it never invents another target forward.
use super::*;
use super::forward_plan::SavedHidden;
use mlx_native::{DType, MlxBuffer};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Reference {
    pub id: u64,
    pub receipt: acquisition::FileBinding,
    pub prefix_sha256: String,
    pub tokens: usize,
    pub has_spec_hidden: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    id: u64,
    identity: Identity,
    acquisition: acquisition::Acquisition,
    prefix_sha256: String,
    tokens: usize,
    vocab: usize,
    hidden_width: usize,
    position: usize,
    plan: acquisition::FileBinding,
    normal: acquisition::FileBinding,
    spec_hidden: Option<SpecWitness>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecWitness {
    plan: acquisition::FileBinding,
    trace: acquisition::FileBinding,
}

#[derive(Serialize)]
struct Plan<'a> {
    schema: &'static str,
    id: u64,
    identity: &'a Identity,
    acquisition: &'a acquisition::Acquisition,
    prefix_sha256: &'a str,
    tokens: usize,
    expected: &'a [ExpectedTensor],
    native_prefill_required: bool,
}

pub(super) fn prefix_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"hf2q.c2.checkpoint-prefix.le-u32.v1\0");
    digest.update((tokens.len() as u64).to_le_bytes());
    for token in tokens { digest.update(token.to_le_bytes()); }
    format!("{:x}", digest.finalize())
}

pub(super) struct Prepared {
    id: u64,
    root: PathBuf,
    scope_root: PathBuf,
    acquisition: acquisition::Acquisition,
    budget: Rc<RefCell<super::matrix_plan::Budget>>,
    prefix: String,
    tokens: usize,
    vocab: usize,
    hidden_width: usize,
    position: usize,
    plan: acquisition::FileBinding,
    store: Store,
}

fn extent(agent: &str, stage: Stage, segment: u32, position: usize, width: usize) -> ExpectedTensor {
    ExpectedTensor { at: Coordinate { logical_step: 0, layer: 0, stage, agent: agent.into(),
        logical_row_start: position as u64, segment }, scalar: Scalar::F32, shape: vec![1, width] }
}

impl Prepared {
    pub(super) fn begin(fixture: &Fixture, prompt: &[u32]) -> Result<Self> {
        let budget = fixture.matrix_budget()?;
        let LoadedModel::Qwen35(qwen) = fixture.loaded.as_ref() else { unreachable!() };
        let vocab = qwen.vocab_size;
        let hidden_width = qwen.hidden_size;
        let graft = qwen.model.kv_graft.as_ref().map_or(0, |graft| graft.bank.n_slots as usize);
        let position = prompt.len().checked_add(graft).and_then(|n| n.checked_sub(1))
            .context("checkpoint witness has no requested final row")?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        ensure!(id != 0, "checkpoint witness identity overflow");
        let agent = format!("checkpoint-{id}/slot0");
        let expected = vec![extent(&agent, Stage::Logits, 0, position, vocab),
            extent(&agent, Stage::LayerOutput, 0, position, hidden_width)];
        let bytes = vocab.checked_add(hidden_width).and_then(|n| n.checked_mul(4))
            .context("checkpoint witness grant overflow")?;
        budget.borrow_mut().reserve_store(expected.len(), u64::try_from(bytes)?)?;
        let root = fixture.packet.output.join(format!("checkpoint-witness-{id:06}"));
        let acquisition = fixture.observed(&fixture.packet.output)?;
        let prefix = prefix_digest(prompt);
        let plan_path = fixture.packet.output.join(format!("checkpoint-witness-{id:06}-plan.json"));
        let plan_sha256 = super::metadata::write(&plan_path, budget.as_ref(), &Plan {
            schema: "hf2q.c2.checkpoint-head-plan.v1", id, identity: &fixture.packet.identity,
            acquisition: &acquisition, prefix_sha256: &prefix, tokens: prompt.len(), expected: &expected,
            native_prefill_required: true,
        })?;
        let mut store = Store::new(&root, fixture.packet.identity.clone(), "C2-original-native-checkpoint-head".into(),
            expected, fixture.packet.host_payload_limit, 2).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(acquisition.clone().for_store_root(&root)?).map_err(anyhow::Error::msg)?;
        Ok(Self { id, root, scope_root: fixture.packet.output.clone(), acquisition, budget,
            prefix, tokens: prompt.len(), vocab, hidden_width, position,
            plan: acquisition::FileBinding { path: plan_path, sha256: plan_sha256 }, store })
    }

    pub(super) fn finish(mut self, logits: &[f32], normalized: &[f32],
        spec_hidden: Option<&MlxBuffer>) -> Result<Reference> {
        ensure!(logits.len() == self.vocab && normalized.len() == self.hidden_width,
            "original checkpoint row differs from its pre-native request");
        let agent = format!("checkpoint-{}/slot0", self.id);
        for (stage, values) in [(Stage::Logits, logits), (Stage::LayerOutput, normalized)] {
            super::trace::capture_f32(&mut self.store,
                extent(&agent, stage, 0, self.position, values.len()).at,
                values, vec![1, if stage == Stage::Logits { self.vocab } else { self.hidden_width }])?;
        }
        let normal = self.store.finish().map_err(anyhow::Error::msg)?;
        ensure!(normal.coverage.errors.is_empty() && normal.coverage.expected.len() == normal.coverage.captured.len(),
            "Incomplete: original checkpoint head witness");
        let normal_path = self.root.join("trace.json");
        let normal_sha256 = super::metadata::hash_file(&normal_path, self.budget.as_ref())?;
        let spec_binding = if let Some(buffer) = spec_hidden {
            ensure!(buffer.dtype() == DType::F32 && buffer.element_count() == self.hidden_width,
                "original speculative boundary has unexpected requested representation");
            // Presence is the production MTP state variant; geometry is the
            // model's requested hidden width, never a readback-derived extent.
            let expected = vec![extent(&agent, Stage::LayerOutput, 1, self.position, self.hidden_width)];
            self.budget.borrow_mut().reserve_store(1, u64::try_from(self.hidden_width.checked_mul(4)
                .context("speculative witness byte count overflow")?)?)?;
            let spec_root = self.root.join("spec");
            let plan_path = self.root.join("spec-plan.json");
            let plan_sha256 = super::metadata::write(&plan_path, self.budget.as_ref(), &Plan {
                schema: "hf2q.c2.checkpoint-spec-plan.v1", id: self.id, identity: &normal.identity,
                acquisition: &self.acquisition, prefix_sha256: &self.prefix, tokens: self.tokens,
                expected: &expected, native_prefill_required: false,
            })?;
            let mut store = Store::new(&spec_root, normal.identity.clone(), "C2-original-spec-boundary".into(),
                expected, u64::try_from(self.hidden_width)? * 4, 1).map_err(anyhow::Error::msg)?;
            store.bind_acquisition(self.acquisition.clone().for_store_root(&spec_root)?).map_err(anyhow::Error::msg)?;
            let values = buffer.as_logical_slice::<f32>()?;
            super::trace::capture_f32(&mut store, extent(&agent, Stage::LayerOutput, 1, self.position, self.hidden_width).at,
                values, vec![1, self.hidden_width])?;
            let trace = store.finish().map_err(anyhow::Error::msg)?;
            ensure!(trace.coverage.errors.is_empty() && trace.coverage.captured.len() == 1,
                "Incomplete: original speculative hidden witness");
            let path = spec_root.join("trace.json");
            Some(SpecWitness { plan: acquisition::FileBinding { path: plan_path, sha256: plan_sha256 },
                trace: acquisition::FileBinding { sha256: super::metadata::hash_file(&path, self.budget.as_ref())?, path } })
        } else { None };
        let receipt = Receipt { schema: "hf2q.c2.checkpoint-head-witness.v1".into(), id: self.id,
            identity: normal.identity, acquisition: self.acquisition, prefix_sha256: self.prefix.clone(),
            tokens: self.tokens, vocab: self.vocab, hidden_width: self.hidden_width, position: self.position,
            plan: self.plan, normal: acquisition::FileBinding { path: normal_path, sha256: normal_sha256 },
            spec_hidden: spec_binding };
        let path = self.root.join("receipt.json");
        let sha256 = super::metadata::write(&path, self.budget.as_ref(), &receipt)?;
        ensure!(path.starts_with(&self.scope_root), "checkpoint receipt escaped acquisition scope");
        Ok(Reference { id: self.id, receipt: acquisition::FileBinding { path, sha256 }, prefix_sha256: self.prefix,
            tokens: self.tokens, has_spec_hidden: receipt.spec_hidden.is_some() })
    }
}

impl Reference {
    fn receipt(&self, fixture: &Fixture, prompt: &[u32]) -> Result<Receipt> {
        let raw = acquisition::checked_file(&self.receipt, 1024 * 1024)?;
        let receipt: Receipt = serde_json::from_slice(&raw)?;
        let observed = fixture.observed(&fixture.packet.output)?;
        ensure!(receipt.schema == "hf2q.c2.checkpoint-head-witness.v1" && receipt.id == self.id
            && receipt.identity == fixture.packet.identity
            && super::registered_plan::same_acquisition(&receipt.acquisition, &observed)
            && receipt.prefix_sha256 == prefix_digest(prompt) && receipt.prefix_sha256 == self.prefix_sha256
            && receipt.tokens == prompt.len() && self.tokens == prompt.len()
            && receipt.spec_hidden.is_some() == self.has_spec_hidden,
            "checkpoint witness belongs to another source/acquisition/prefix");
        acquisition::checked_file(&receipt.plan, 1024 * 1024)?;
        if let Some(spec) = &receipt.spec_hidden { acquisition::checked_file(&spec.plan, 1024 * 1024)?; }
        Ok(receipt)
    }
    pub(super) fn planned_mode(&self, fixture: &Fixture, prompt: &[u32], has_cached_spec: bool) -> Result<SavedHidden> {
        let receipt = self.receipt(fixture, prompt)?;
        if has_cached_spec {
            ensure!(receipt.spec_hidden.is_some(), "equality selected spec state without its original hidden witness");
            Ok(SavedHidden::ConsumedSpeculative)
        } else { Ok(SavedHidden::OriginalNormalizedHead) }
    }
    pub(super) fn check_values(&self, fixture: &Fixture, prompt: &[u32], mode: SavedHidden,
        logits: &[f32], hidden: &[f32]) -> Result<()> {
        let receipt = self.receipt(fixture, prompt)?;
        check_row(&receipt.normal, Stage::Logits, 0, receipt.vocab, logits)?;
        match mode {
            SavedHidden::OriginalNormalizedHead => check_row(&receipt.normal, Stage::LayerOutput, 0, receipt.hidden_width, hidden),
            SavedHidden::ConsumedSpeculative => check_row(&receipt.spec_hidden.as_ref()
                .context("original speculative witness absent")?.trace, Stage::LayerOutput, 1, receipt.hidden_width, hidden),
        }
    }
}

fn check_row(binding: &acquisition::FileBinding, stage: Stage, segment: u32, width: usize,
    actual: &[f32]) -> Result<()> {
    ensure!(actual.len() == width, "consumed checkpoint row has wrong requested width");
    let raw = acquisition::checked_file(binding, 1024 * 1024)?;
    let trace: Trace = serde_json::from_slice(&raw)?;
    ensure!(trace.coverage.errors.is_empty() && trace.coverage.expected.len() == trace.coverage.captured.len(),
        "original checkpoint trace is incomplete");
    let mut rows = trace.coverage.captured.iter().filter(|row| row.at.stage == stage && row.at.segment == segment);
    let row = rows.next().context("original checkpoint row missing")?;
    ensure!(rows.next().is_none() && row.scalar == Scalar::F32 && row.shape == [1, width],
        "original checkpoint row is ambiguous or has wrong representation");
    let root = binding.path.parent().context("checkpoint trace has no private root")?;
    let path = root.join(&row.file);
    ensure!(path.canonicalize()?.starts_with(root), "checkpoint tensor escaped its bound trace root");
    let raw = acquisition::checked_file(&acquisition::FileBinding { path, sha256: row.sha256.clone() },
        u64::try_from(width.checked_mul(4).context("checkpoint row byte count overflow")?)?)?;
    ensure!(raw.len() == width * 4 && raw.chunks_exact(4).zip(actual)
        .all(|(bytes, value)| bytes == value.to_bits().to_le_bytes()),
        "DivergedExactInvariant: equality did not consume its original checkpoint row");
    Ok(())
}
