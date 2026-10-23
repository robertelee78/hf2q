use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{DeclaredBytes, Error, FrozenIdentity};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Family {
    Qwen38,
    Qwen36,
    Gemma4,
    Deepseek4,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenShape {
    pub identity: FrozenIdentity,
    pub family: Family,
    pub slots: u8,
    pub context_per_slot: u64,
    pub result_cap: u64,
    pub dispatch_threads: u8,
    pub vision: bool,
    pub drafter: bool,
    pub graft: bool,
    pub persistence: bool,
}

impl FrozenShape {
    pub fn validate(&self) -> Result<(), Error> {
        self.identity.validate()?;
        if self.slots != 4
            || self.context_per_slot == 0
            || self.result_cap == 0
            || self.result_cap > self.context_per_slot
            || !(1..=64).contains(&self.dispatch_threads)
            || (self.family == Family::Deepseek4 && self.vision)
        {
            return Err(Error::Identity("invalid four-slot frozen shape".into()));
        }
        Ok(())
    }

    /// Closed schema. Optional features remain present as explicit zero or
    /// unknown; callers cannot remove keys by leaving a feature out of a list.
    pub fn required_keys(&self) -> Result<BTreeSet<CategoryKey>, Error> {
        self.validate()?;
        use Category::*;
        let mut keys = BTreeSet::new();
        for kind in [
            WeightResources,
            ExpandedWeights,
            LoadStaging,
            HostBaseline,
            Projector,
            VisionBuffers,
            GraftBank,
            Drafter,
            PersistentScratch,
            CapturedResources,
            QueuedResources,
            CompletedResources,
            PendingResidencyRemoval,
            SubmissionInputs,
            ServiceQueues,
            PersistBusy,
            PersistPending,
            PersistConstructing,
            PersistHydration,
            PersistSerialization,
        ] {
            keys.insert(CategoryKey {
                owner: Owner::Process,
                kind,
            });
        }
        let mut owners: Vec<_> = (0..self.slots).map(Owner::Slot).collect();
        if self.family == Family::Deepseek4 {
            owners.push(Owner::Boot);
        }
        for owner in owners {
            for kind in [
                LiveCache,
                RecurrentState,
                ConvolutionState,
                MtpState,
                IndexerState,
                CompressorState,
                LiveLogits,
                TokenLedger,
                CommittedAnchor,
                PendingAnchor,
                RollbackAnchor,
                AnchorStore,
                GrowthOverlap,
            ] {
                keys.insert(CategoryKey { owner, kind });
            }
        }
        for thread in 0..self.dispatch_threads {
            let pools: &[Category] = match self.family {
                Family::Deepseek4 => &[PrefillPool, SubmissionPool, DecodePool],
                Family::Qwen38 | Family::Qwen36 => &[DecodePool, WeightPool],
                Family::Gemma4 => &[DirectScratch],
            };
            for &kind in pools {
                keys.insert(CategoryKey {
                    owner: Owner::Thread(thread),
                    kind,
                });
            }
        }
        Ok(keys)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Owner {
    Process,
    Slot(u8),
    Boot,
    Thread(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Category {
    WeightResources,
    ExpandedWeights,
    LoadStaging,
    HostBaseline,
    Projector,
    VisionBuffers,
    GraftBank,
    Drafter,
    PersistentScratch,
    CapturedResources,
    QueuedResources,
    CompletedResources,
    PendingResidencyRemoval,
    SubmissionInputs,
    ServiceQueues,
    PersistBusy,
    PersistPending,
    PersistConstructing,
    PersistHydration,
    PersistSerialization,
    LiveCache,
    RecurrentState,
    ConvolutionState,
    MtpState,
    IndexerState,
    CompressorState,
    LiveLogits,
    TokenLedger,
    CommittedAnchor,
    PendingAnchor,
    RollbackAnchor,
    AnchorStore,
    GrowthOverlap,
    PrefillPool,
    SubmissionPool,
    DecodePool,
    WeightPool,
    DirectScratch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoryKey {
    pub owner: Owner,
    pub kind: Category,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Entry {
    Bound {
        bytes: DeclaredBytes,
        provenance: String,
        formula_revision: String,
    },
    ProvenZero {
        reason: String,
        configuration_evidence: String,
    },
    Unknown {
        reason: String,
    },
}

impl Entry {
    fn bytes(&self, key: CategoryKey) -> Result<DeclaredBytes, Error> {
        match self {
            Self::Bound {
                bytes,
                provenance,
                formula_revision,
            } if !provenance.trim().is_empty() && !formula_revision.trim().is_empty() => {
                bytes.validate()?;
                Ok(*bytes)
            }
            Self::ProvenZero {
                reason,
                configuration_evidence,
            } if !reason.trim().is_empty() && !configuration_evidence.trim().is_empty() => {
                Ok(DeclaredBytes::ZERO)
            }
            Self::Unknown { reason } => Err(Error::Unknown(format!("{key:?}: {reason}"))),
            _ => Err(Error::Category(format!(
                "{key:?}: missing declared provenance"
            ))),
        }
    }
}

/// Entries are unique, disjoint accounting charges, not overlapping owner
/// labels. A shared allocation belongs to one charge; aliases go in the trace.
pub struct Inventory {
    shape: FrozenShape,
    entries: BTreeMap<CategoryKey, Entry>,
}

impl Inventory {
    /// Every real packet begins incomplete. Feature disabling does not infer
    /// zero: the caller must explicitly account for retained prior resources.
    pub fn unknown(shape: FrozenShape) -> Result<Self, Error> {
        let entries = shape
            .required_keys()?
            .into_iter()
            .map(|key| {
                (
                    key,
                    Entry::Unknown {
                        reason: "required bound not established".into(),
                    },
                )
            })
            .collect();
        Ok(Self { shape, entries })
    }

    /// Imported rows must match the exact schema, including duplicate detection.
    pub fn from_entries(
        shape: FrozenShape,
        rows: Vec<(CategoryKey, Entry)>,
    ) -> Result<Self, Error> {
        let required = shape.required_keys()?;
        let mut entries = BTreeMap::new();
        for (key, entry) in rows {
            if !required.contains(&key) || entries.insert(key, entry).is_some() {
                return Err(Error::Category(format!("unexpected or duplicate {key:?}")));
            }
        }
        if entries.len() != required.len() {
            return Err(Error::Category("missing required category".into()));
        }
        Ok(Self { shape, entries })
    }

    pub fn declare(&mut self, key: CategoryKey, entry: Entry) -> Result<(), Error> {
        let target = self
            .entries
            .get_mut(&key)
            .ok_or_else(|| Error::Category(format!("unexpected {key:?}")))?;
        *target = entry;
        Ok(())
    }

    pub fn require_shape(&self, expected: &FrozenShape) -> Result<(), Error> {
        expected.validate()?;
        if &self.shape != expected {
            return Err(Error::Identity(
                "packet differs from frozen workload".into(),
            ));
        }
        Ok(())
    }

    pub fn entries(&self) -> Vec<(CategoryKey, Entry)> {
        self.entries
            .iter()
            .map(|(&key, entry)| (key, entry.clone()))
            .collect()
    }

    pub fn total(&self) -> Result<DeclaredBytes, Error> {
        self.entries
            .iter()
            .try_fold(DeclaredBytes::ZERO, |sum, (&key, entry)| {
                sum.checked_add(entry.bytes(key)?)
            })
    }
}
