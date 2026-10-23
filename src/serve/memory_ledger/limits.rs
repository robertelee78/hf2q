use serde::{Deserialize, Serialize};

use super::{Error, Family};

pub const MIN_RESERVE: u64 = 12 * (1 << 30);

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

/// Declared inputs only. A Bound competitor value does not attest enumeration,
/// freshness, privilege, metric semantics or real physical availability.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub physical: Ceiling,
    pub reserve: u64,
    pub competition: Ceiling,
    pub family: Ceiling,
    pub working_set: Ceiling,
}

#[derive(Clone, Debug, Serialize)]
pub struct Projection {
    pub bytes: DeclaredBytes,
    pub host_cap_after_competition: u64,
    pub family_cap: Option<u64>,
    pub working_set_cap: u64,
}

impl Limits {
    pub fn require_family(&self, family: Family) -> Result<(), Error> {
        let bound = self.family.bound(true)?;
        if family == Family::Deepseek4 {
            match bound {
                Some(cap) if cap <= 116 * (1 << 30) => {}
                _ => {
                    return Err(Error::Unknown(
                        "DeepSeek requires its existing <=116 GiB family ceiling".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    pub fn project(&self, bytes: DeclaredBytes) -> Result<Projection, Error> {
        bytes.validate()?;
        if self.reserve < MIN_RESERVE {
            return Err(Error::Limit("system reserve below 12 GiB"));
        }
        let physical = self
            .physical
            .bound(false)?
            .ok_or(Error::Unknown("physical".into()))?;
        let competition = self
            .competition
            .bound(false)?
            .ok_or(Error::Unknown("competition".into()))?;
        let family = self.family.bound(true)?;
        let working_set = self
            .working_set
            .bound(false)?
            .ok_or(Error::Unknown("working set".into()))?;
        let charged = bytes
            .owned
            .checked_add(competition)
            .ok_or(Error::Overflow)?
            .checked_add(self.reserve)
            .ok_or(Error::Overflow)?;
        if charged > physical {
            return Err(Error::Limit("host including reserve and competition"));
        }
        if family.is_some_and(|cap| bytes.owned > cap) {
            return Err(Error::Limit("family process ceiling"));
        }
        if bytes.metal > working_set {
            return Err(Error::Limit("declared Metal working-set policy"));
        }
        Ok(Projection {
            bytes,
            host_cap_after_competition: physical - self.reserve - competition,
            family_cap: family,
            working_set_cap: working_set,
        })
    }
}
