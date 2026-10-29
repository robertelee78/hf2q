//! Selected-only pre-clone work quotas. Not allocator usage or physical credit.
use anyhow::{ensure, Context, Result};
use mlx_native::gguf::{GgufFile, GgufHeaderLimits, MetadataValue};
use serde::Serialize;

const MIB: u64 = 1024 * 1024;
const MAX_TOKENS: usize = 262_144;
const MAX_MERGES: usize = 524_288;
const MAX_UTF8: u64 = 32 * MIB;

pub(super) fn limits() -> GgufHeaderLimits {
    GgufHeaderLimits {
        max_read_bytes: 64 * MIB,
        max_declared_allocation_bytes: 64 * MIB,
        max_metadata_entries: 4096,
        max_tensors: 4096,
        max_string_bytes: MIB,
        max_total_array_elements: 1_000_000,
        max_nesting_depth: 4,
    }
}

pub(crate) fn check_origin(external: bool) -> Result<()> {
    ensure!(
        !external,
        "Q5 external tokenizer startup envelope Unknown: override refused"
    );
    Ok(())
}

#[derive(Debug, Serialize)]
#[serde(tag = "status")]
pub(super) enum Facts {
    Unknown {
        reason: &'static str,
    },
    BoundedEmbeddedMetadata {
        header_read_bytes: u64,
        header_declared_reservation_bytes: u64,
        header_read_limit: u64,
        header_declared_limit: u64,
        token_count: usize,
        merge_count: usize,
        aggregate_utf8_bytes: u64,
        token_count_limit: usize,
        merge_count_limit: usize,
        aggregate_utf8_limit: u64,
        tokenizer_override_join: &'static str,
        authority: &'static str,
    },
}

fn array<'a>(header: &'a GgufFile, key: &str) -> Result<&'a [MetadataValue]> {
    match header.metadata(key) {
        Some(MetadataValue::Array(values)) => Ok(values),
        _ => anyhow::bail!("Q5 tokenizer requires typed array {key}"),
    }
}
fn add_utf8(total: u64, next: usize) -> Result<u64> {
    let total = total
        .checked_add(u64::try_from(next)?)
        .context("Q5 tokenizer UTF8 overflow")?;
    ensure!(total <= MAX_UTF8, "Q5 tokenizer aggregate UTF8 quota");
    Ok(total)
}
// Borrows every string; no String/Vec clone or tokenizer construction occurs.
fn validate(vocab: u32, tokens: &[MetadataValue], merges: &[MetadataValue]) -> Result<u64> {
    ensure!(
        !tokens.is_empty() && tokens.len() <= MAX_TOKENS,
        "Q5 tokenizer token count quota"
    );
    ensure!(
        tokens.len() == usize::try_from(vocab)?,
        "Q5 tokenizer token count != strict config vocab"
    );
    ensure!(merges.len() <= MAX_MERGES, "Q5 tokenizer merge count quota");
    let mut total = 0;
    for value in tokens.iter().chain(merges) {
        let MetadataValue::String(value) = value else {
            anyhow::bail!("Q5 tokenizer non-string entry")
        };
        total = add_utf8(total, value.len())?;
    }
    for value in merges {
        let MetadataValue::String(value) = value else {
            unreachable!()
        };
        ensure!(
            value.split_once(' ').is_some(),
            "Q5 tokenizer merge lacks separator"
        );
    }
    Ok(total)
}

pub(super) fn inspect(header: &GgufFile) -> Result<Facts> {
    use crate::inference::models::deepseek4::Deepseek4Model;
    // Header-only refusal controls need not manufacture a valid model config.
    // Unknown is never an admission witness.
    let Ok(cfg) = Deepseek4Model::load_config_only(header) else {
        return Ok(Facts::Unknown {
            reason: "ActualStrictConfigUnavailable_NoTokenizerEnvelope",
        });
    };
    ensure!(
        header.metadata_string("tokenizer.ggml.model") == Some("gpt2")
            && header.metadata_string("tokenizer.ggml.pre") == Some("deepseek-v3"),
        "Q5 embedded tokenizer family mismatch"
    );
    let tokens = array(header, "tokenizer.ggml.tokens")?;
    let merges = array(header, "tokenizer.ggml.merges")?;
    let aggregate_utf8_bytes = validate(cfg.vocab_size, tokens, merges)?;
    if let Some(types) = header.metadata("tokenizer.ggml.token_type") {
        let MetadataValue::Array(types) = types else {
            anyhow::bail!("Q5 token_type is not an array")
        };
        ensure!(types.len() == tokens.len(), "Q5 token_type count mismatch");
        ensure!(
            types.iter().all(|v| matches!(
                v,
                MetadataValue::Int32(_)
                    | MetadataValue::Uint32(_)
                    | MetadataValue::Int16(_)
                    | MetadataValue::Uint16(_)
                    | MetadataValue::Int8(_)
                    | MetadataValue::Uint8(_)
            )),
            "Q5 token_type non-integer entry"
        );
    }
    let usage = header
        .bounded_header_usage()
        .context("Q5 bounded header usage unavailable")?;
    Ok(Facts::BoundedEmbeddedMetadata {
        header_read_bytes: usage.read_bytes,
        header_declared_reservation_bytes: usage.declared_allocation_bytes,
        header_read_limit: 64 * MIB, header_declared_limit: 64 * MIB,
        token_count: tokens.len(), merge_count: merges.len(), aggregate_utf8_bytes,
        token_count_limit: MAX_TOKENS, merge_count_limit: MAX_MERGES, aggregate_utf8_limit: MAX_UTF8,
        tokenizer_override_join: "HeaderFactsOnly_ActualLoadChecksOverride_DryPreflightUnknown_NoPermit",
        authority: "SourceWorkQuotaOnly_EmbeddedMetadata_OverrideMustBeJoined_NoAllocatorOrPhysicalBound_NoCPUCellReduction",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(v: &str) -> MetadataValue {
        MetadataValue::String(v.into())
    }
    #[test]
    fn borrowed_tokenizer_positive_and_exact_vocab() {
        assert_eq!(validate(2, &[s("a"), s("b")], &[s("a b")]).unwrap(), 5);
        assert!(validate(3, &[s("a"), s("b")], &[]).is_err());
    }
    #[test]
    fn typed_arrays_and_merges_are_required() {
        assert!(validate(1, &[MetadataValue::Uint32(1)], &[]).is_err());
        assert!(validate(1, &[s("a")], &[MetadataValue::Uint32(1)]).is_err());
        assert!(validate(1, &[s("a")], &[s("ab")]).is_err());
    }
    #[test]
    fn aggregate_count_and_overflow_refuse_without_cloning_payload() {
        assert!(add_utf8(MAX_UTF8, 1).is_err());
        assert!(add_utf8(u64::MAX, 1).is_err());
        assert!(validate(0, &[], &[]).is_err());
        assert!(validate(1, &[s("a")], &vec![s("a b"); MAX_MERGES + 1]).is_err());
    }
    #[test]
    fn actual_override_is_unknown_not_embedded() {
        assert!(check_origin(false).is_ok());
        assert!(check_origin(true).is_err());
    }
}
