//! Host inputs are bound at the independent API boundary. GPU inputs retain
//! their actual descriptor and checked-completion readback, never placeholders.
use super::*;
use super::{registered_plan::Bundle, trace::{Forward, Observations}};
use mlx_native::{DType, MlxBuffer};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum InputBinding {
    Host { sha256: String },
    GpuCheckedCompletion { buffer_identity: u64, buffer_tokens: usize, first: usize, rows: usize },
}
fn hash_tokens(length: usize, tokens: impl IntoIterator<Item = u32>) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"hf2q.c2.native-input.le-u32.v1\0");
    digest.update(u64::try_from(length)?.to_le_bytes());
    let mut count = 0_usize;
    for token in tokens {
        count = count.checked_add(1).context("native input count overflow")?;
        ensure!(count <= length, "native host input exceeds its requested rows");
        digest.update(token.to_le_bytes());
    }
    ensure!(count == length, "native host input omitted requested rows");
    Ok(format!("{:x}", digest.finalize()))
}
impl InputBinding {
    pub(crate) fn host(tokens: &[u32]) -> Result<Self> {
        Self::host_rows(tokens.len(), tokens.iter().copied())
    }
    pub(crate) fn host_rows(rows: usize, tokens: impl IntoIterator<Item = u32>) -> Result<Self> {
        ensure!(rows > 0, "native host input is empty");
        Ok(Self::Host { sha256: hash_tokens(rows, tokens)? })
    }
    pub(crate) fn gpu(buffer: &MlxBuffer, first: usize, rows: usize) -> Result<Self> {
        ensure!(buffer.dtype() == DType::U32, "native GPU token input has another representation");
        let value = Self::GpuCheckedCompletion { buffer_identity: buffer as *const _ as usize as u64,
            buffer_tokens: buffer.element_count(), first, rows };
        value.geometry(rows)?;
        Ok(value)
    }
    pub(super) fn geometry(&self, length: usize) -> Result<()> {
        ensure!(length > 0, "native input has no requested rows");
        match self {
            Self::Host { sha256 } => ensure!(sha256.len() == 64 && sha256.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)), "invalid requested host input digest"),
            Self::GpuCheckedCompletion { buffer_identity, buffer_tokens, first, rows } => {
                ensure!(*buffer_identity != 0 && *rows == length
                    && first.checked_add(*rows).is_some_and(|end| end <= *buffer_tokens),
                    "native GPU input range differs from its immutable API descriptor");
            }
        }
        Ok(())
    }
    pub(super) fn check(&self, tokens: &[u32]) -> Result<String> {
        self.geometry(tokens.len())?;
        let actual = hash_tokens(tokens.len(), tokens.iter().copied())?;
        if let Self::Host { sha256 } = self {
            ensure!(actual == *sha256, "actual consumed tokens differ from the independently requested host input");
        }
        Ok(actual)
    }
}

#[derive(Serialize)]
struct Record<'a> {
    sequence: u64,
    slot: u32,
    start: usize,
    source: &'a InputBinding,
    actual_tokens: &'a [u32],
    actual_sha256: &'a str,
}
struct Records<'a> { forwards: &'a [Forward], plan: &'a Bundle }
impl Serialize for Records<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeSeq};
        let mut sequence = serializer.serialize_seq(Some(self.forwards.len()))?;
        for forward in self.forwards {
            let id = forward.planned_sequence.ok_or_else(|| S::Error::custom("native input source identity absent"))?;
            let source = self.plan.inputs.get(&id).ok_or_else(|| S::Error::custom("native input was not requested"))?;
            sequence.serialize_element(&Record { sequence: id, slot: forward.slot, start: forward.start,
                source, actual_tokens: &forward.input,
                actual_sha256: forward.input_sha256.as_deref().ok_or_else(|| S::Error::custom("checked input digest absent"))? })?;
        }
        sequence.end()
    }
}
#[derive(Serialize)]
struct Index<'a> {
    schema: &'static str,
    context: super::registration::HeaderRef<'a>,
    execution: &'a acquisition::FileBinding,
    records: Records<'a>,
}
pub(super) fn write(context: super::registration::HeaderRef<'_>, scope_root: &Path,
    budget: &std::rc::Rc<std::cell::RefCell<super::matrix_plan::Budget>>, observations: &Observations,
    plan: &Bundle, root: &Path) -> Result<acquisition::FileBinding> {
    ensure!(observations.forwards.len() == plan.inputs.len(), "native input record coverage differs from required calls");
    let mut seen = BTreeSet::new();
    for forward in &observations.forwards {
        let id = forward.planned_sequence.context("native input has no source sequence")?;
        ensure!(seen.insert(id), "native input source sequence repeated");
        let source = plan.inputs.get(&id).context("native input source was not registered")?;
        plan.forwards.get(&id).context("native input lacks its requested geometry")?.check(forward)?;
        let digest = source.check(&forward.input)?;
        ensure!(forward.input_sha256.as_deref() == Some(digest.as_str()),
            "native input changed after its checked-completion observation");
    }
    let path = root.join("native-inputs.json");
    let sha256 = super::metadata::write(&path, budget.as_ref(), &Index {
        schema: "hf2q.c2.native-input-records.v1", context,
        execution: &plan.execution, records: Records { forwards: &observations.forwards, plan },
    })?;
    Ok(acquisition::FileBinding { path: path.strip_prefix(scope_root)?.into(), sha256 })
}

#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn c2_host_input_binding_rejects_same_length_token_substitution() {
        let input = InputBinding::host(&[1, 2, 3]).unwrap();
        assert!(input.check(&[1, 8, 3]).is_err());
        assert!(input.check(&[1, 2]).is_err());
    }
}
