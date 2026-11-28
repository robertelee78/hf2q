//! Exact raw coordinates come from requested projection geometry before native
//! execution. Returned tensor lengths may validate that plan, never create it.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum SavedHidden { ConsumedSpeculative, OriginalNormalizedHead }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum Origin {
    Native,
    SavedCheckpoint { witness: u64, hidden: SavedHidden },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ForwardRequest {
    pub sequence: u64,
    pub slot: u32,
    pub start: usize,
    pub input_rows: usize,
    pub projected_rows: usize,
    pub vocab: usize,
    pub hidden_width: usize,
    pub origin: Origin,
}

impl ForwardRequest {
    pub(super) fn first_row(self) -> Result<usize> {
        ensure!(self.input_rows > 0 && self.projected_rows > 0
            && self.projected_rows <= self.input_rows && self.vocab > 0 && self.hidden_width > 0,
            "invalid source-requested C2 projection geometry");
        if let Origin::SavedCheckpoint { witness, .. } = self.origin {
            ensure!(witness != 0 && self.input_rows == 1 && self.projected_rows == 1,
                "saved checkpoint consumption requires its original one-row witness");
        }
        self.start.checked_add(self.input_rows - self.projected_rows)
            .context("requested C2 projection position overflow")
    }

    pub(super) fn logical_bytes(self) -> Result<u64> {
        self.first_row()?;
        let values = self.vocab.checked_add(self.hidden_width)
            .and_then(|n| n.checked_mul(self.projected_rows))
            .and_then(|n| n.checked_mul(4))
            .context("requested C2 projection byte count overflow")?;
        Ok(u64::try_from(values)?)
    }

    pub(super) fn expected(self, agent: &str) -> Result<Vec<ExpectedTensor>> {
        let first = self.first_row()?;
        self.logical_bytes()?;
        let count = self.projected_rows.checked_mul(2).context("projection record count overflow")?;
        ensure!(count <= 65536, "source projection exceeds strict Q1 record grant");
        let mut expected = Vec::with_capacity(count);
        for (stage, width) in [(Stage::Logits, self.vocab), (Stage::LayerOutput, self.hidden_width)] {
            for row in 0..self.projected_rows {
                expected.push(ExpectedTensor { at: Coordinate { logical_step: self.sequence,
                    layer: 0, stage, agent: agent.into(),
                    logical_row_start: u64::try_from(first.checked_add(row)
                        .context("requested C2 row position overflow")?)?, segment: 0 },
                    scalar: Scalar::F32, shape: vec![1, width] });
            }
        }
        Ok(expected)
    }

    pub(super) fn check(self, observed: &super::trace::Forward) -> Result<()> {
        self.first_row()?;
        let logits = self.projected_rows.checked_mul(self.vocab)
            .context("requested logit extent overflow")?;
        let hidden = self.projected_rows.checked_mul(self.hidden_width)
            .context("requested normalized-hidden extent overflow")?;
        ensure!(observed.slot == self.slot && observed.start == self.start
            && observed.input.len() == self.input_rows && observed.vocab == self.vocab
            && observed.hidden_width == self.hidden_width
            && observed.logits.len() == logits && observed.hidden.len() == hidden,
            "Incomplete: actual forward differs from its pre-call requested projection");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecisionRequest {
    pub semantic_index: u64,
    pub physical_position: u64,
    pub vocab: usize,
    pub hidden_width: usize,
}

impl DecisionRequest {
    pub(super) fn raw_expected(self, agent: &str) -> ExpectedTensor {
        ExpectedTensor { at: Coordinate { logical_step: self.semantic_index, layer: 0,
            stage: Stage::SpecCanonicalIds, agent: agent.into(),
            logical_row_start: self.semantic_index, segment: 0 }, scalar: Scalar::U32, shape: vec![1] }
    }
    pub(super) fn canonical_expected(self, agent: &str) -> Result<[ExpectedTensor; 3]> {
        ensure!(self.vocab > 0 && self.hidden_width > 0, "invalid canonical requested geometry");
        let at = |stage| Coordinate { logical_step: self.semantic_index, layer: 0,
            stage, agent: agent.into(), logical_row_start: self.physical_position, segment: 0 };
        Ok([
            ExpectedTensor { at: at(Stage::Logits), scalar: Scalar::F32, shape: vec![1, self.vocab] },
            ExpectedTensor { at: at(Stage::LayerOutput), scalar: Scalar::F32, shape: vec![1, self.hidden_width] },
            ExpectedTensor { at: at(Stage::SpecCanonicalIds), scalar: Scalar::U32, shape: vec![1] },
        ])
    }
}

#[test]
fn c2_forward_plan_rejects_symmetric_missing_verifier_tail_or_hidden_row() {
    let requested = ForwardRequest { sequence: 0, slot: 0, start: 10, input_rows: 3,
        projected_rows: 2, vocab: 4, hidden_width: 3, origin: Origin::Native };
    let registered = requested.expected("case/slot0").unwrap();
    for _arm in ["cold", "resumed"] {
        let mut observed = super::trace::Forward { slot: 0, start: 10, input: vec![1, 2, 3],
            route: "synthetic CPU geometry control".into(), logits: vec![0; 8], hidden: vec![0; 6],
            vocab: 4, hidden_width: 3, planned_sequence: None, input_sha256: None };
        requested.check(&observed).unwrap();
        observed.logits.truncate(4);
        assert!(requested.check(&observed).is_err(), "even an unaccepted verifier tail is required");
        observed.logits.resize(8, 0);
        observed.hidden.truncate(3);
        assert!(requested.check(&observed).is_err(), "all requested normalized-hidden rows are required");
        // Registration was frozen before either observation existed.
        assert!(registered.iter().any(|extent| extent.at.stage == Stage::Logits
            && extent.at.logical_row_start == 12 && extent.shape == [1, 4]));
    }
}

#[test]
fn c2_forward_plan_saved_consumption_cannot_invent_a_checkpoint_or_extra_rows() {
    let mut requested = ForwardRequest { sequence: 0, slot: 0, start: 10, input_rows: 1,
        projected_rows: 1, vocab: 4, hidden_width: 3,
        origin: Origin::SavedCheckpoint { witness: 0, hidden: SavedHidden::OriginalNormalizedHead } };
    assert!(requested.expected("case/slot0").is_err());
    requested.origin = Origin::SavedCheckpoint { witness: 7, hidden: SavedHidden::OriginalNormalizedHead };
    requested.expected("case/slot0").unwrap();
    requested.input_rows = 2;
    requested.projected_rows = 2;
    assert!(requested.expected("case/slot0").is_err());
}
