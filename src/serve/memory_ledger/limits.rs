use serde::{Deserialize, Serialize};

use super::{Error, Family};

/// owned includes shared Metal bytes exactly once. metal is a subset, not an
/// additional host charge; CPU-only buffers have metal=0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredBytes {
    pub owned: u64,
    pub metal: u64,
}

impl DeclaredBytes {
    pub const ZERO: Self = Self { owned: 0, metal: 0 };

    pub fn validate(self) -> Result<(), Error> {
        if self.metal > self.owned {
            return Err(Error::Category(
                "Metal bytes must be a subset of owned bytes".into(),
            ));
        }
        Ok(())
    }

    pub fn checked_add(self, other: Self) -> Result<Self, Error> {
        self.validate()?;
        other.validate()?;
        Ok(Self {
            owned: self.owned.checked_add(other.owned).ok_or(Error::Overflow)?,
            metal: self.metal.checked_add(other.metal).ok_or(Error::Overflow)?,
        })
    }

    pub fn covers(self, other: Self) -> bool {
        self.owned >= other.owned && self.metal >= other.metal
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Ceiling {
    Bound { bytes: u64, provenance: String },
    NoneWithInventory { provenance: String },
    Unknown { reason: String },
}

impl Ceiling {
    fn bound(&self, optional: bool) -> Result<Option<u64>, Error> {
        match self {
            Self::Bound { bytes, provenance } if !provenance.trim().is_empty() => Ok(Some(*bytes)),
            Self::NoneWithInventory { provenance } if optional && !provenance.trim().is_empty() => {
                Ok(None)
            }
            Self::Unknown { reason } => Err(Error::Unknown(reason.clone())),
            _ => Err(Error::Unknown(
                "missing required bound or provenance".into(),
            )),
        }
    }
}

/// Prospective, caller-declared reaction packet. Times and byte allowances must
/// come from the frozen source/observer/stop evidence; this type attests none.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reaction {
    pub observation_age_bytes: u64,
    pub inflight_overshoot_bytes: u64,
    pub maximum_observation_age_ns: u64,
    pub maximum_stop_latency_ns: u64,
    pub provenance: String,
}
impl Reaction {
    fn bytes(&self) -> Result<u64, Error> {
        if self.observation_age_bytes == 0
            || self.inflight_overshoot_bytes == 0
            || self.maximum_observation_age_ns == 0
            || self.maximum_stop_latency_ns == 0
            || self.provenance.trim().is_empty()
        {
            return Err(Error::Unknown(
                "positive reaction/overshoot packet and provenance required".into(),
            ));
        }
        self.observation_age_bytes
            .checked_add(self.inflight_overshoot_bytes)
            .ok_or(Error::Overflow)
    }
}

/// A recommendation is recorded separately and never enforced as a hardware cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MetalPolicy {
    Enforced { bytes: u64, provenance: String },
    RecommendationOnly { bytes: u64, provenance: String },
    NoneWithInventory { provenance: String },
    Unknown { reason: String },
}
impl MetalPolicy {
    fn cap(&self) -> Result<Option<u64>, Error> {
        match self {
            Self::Enforced { bytes, provenance } if !provenance.trim().is_empty() => {
                Ok(Some(*bytes))
            }
            Self::RecommendationOnly { provenance, .. }
            | Self::NoneWithInventory { provenance }
                if !provenance.trim().is_empty() =>
            {
                Ok(None)
            }
            Self::Unknown { reason } => Err(Error::Unknown(reason.clone())),
            _ => Err(Error::Unknown("Metal policy provenance required".into())),
        }
    }
}

/// Declared inputs only. A Bound competitor value does not attest enumeration,
/// freshness, privilege, metric semantics or real physical availability.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub physical: Ceiling,
    pub reaction: Reaction,
    pub competition: Ceiling,
    pub family: Ceiling,
    pub metal_policy: MetalPolicy,
}

#[derive(Clone, Debug, Serialize)]
pub struct Projection {
    pub bytes: DeclaredBytes,
    pub host_cap_after_competition: u64,
    pub family_cap: Option<u64>,
    pub working_set_cap: Option<u64>,
    pub reaction_bytes: u64,
}

impl Limits {
    pub fn require_family(&self, family: Family) -> Result<(), Error> {
        // Every family follows the same explicit source-inventory rule. No
        // historical DeepSeek 116 GiB ceiling is invented by this module.
        let _ = family;
        self.family.bound(true)?;
        Ok(())
    }

    pub fn project(&self, bytes: DeclaredBytes) -> Result<Projection, Error> {
        bytes.validate()?;
        let reaction = self.reaction.bytes()?;
        let physical = self
            .physical
            .bound(false)?
            .ok_or(Error::Unknown("physical".into()))?;
        let competition = self
            .competition
            .bound(false)?
            .ok_or(Error::Unknown("competition".into()))?;
        let family = self.family.bound(true)?;
        let working_set = self.metal_policy.cap()?;
        let charged = bytes
            .owned
            .checked_add(competition)
            .ok_or(Error::Overflow)?
            .checked_add(reaction)
            .ok_or(Error::Overflow)?;
        if charged > physical {
            return Err(Error::Limit("host including reaction and competition"));
        }
        if family.is_some_and(|cap| bytes.owned > cap) {
            return Err(Error::Limit("family process ceiling"));
        }
        if working_set.is_some_and(|cap| bytes.metal > cap) {
            return Err(Error::Limit("declared Metal working-set policy"));
        }
        Ok(Projection {
            bytes,
            host_cap_after_competition: physical - reaction - competition,
            family_cap: family,
            working_set_cap: working_set,
            reaction_bytes: reaction,
        })
    }
}
