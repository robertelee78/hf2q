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

fn bounded_config_work(header: &GgufFile) -> Result<bool> {
    let Some(layers) = header.metadata_u32("deepseek4.block_count") else {
        return Ok(false);
    };
    ensure!((1..=64).contains(&layers), "Q5 config layer work quota");
    let Some(hidden) = header.metadata_u32("deepseek4.embedding_length") else {
        return Ok(false);
    };
    ensure!(
        hidden > 0 && hidden <= u32::MAX / 4,
        "Q5 config hidden multiplication quota"
    );
    for key in [
        "deepseek4.attention.compress_ratios",
        "deepseek4.swiglu_clamp_exp",
        "deepseek4.swiglu_clamp_shexp",
    ] {
        let Some(MetadataValue::Array(values)) = header.metadata(key) else {
            return Ok(false);
        };
        ensure!(
            values.len() == layers as usize,
            "Q5 config array work quota: {key}"
        );
    }
    Ok(true)
}

pub(super) fn inspect(header: &GgufFile) -> Result<Facts> {
    use crate::inference::models::deepseek4::Deepseek4Model;
    if !bounded_config_work(header)? {
        return Ok(Facts::Unknown {
            reason: "ConfigWorkShapeUnavailable_NoTokenizerEnvelope",
        });
    }
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
    fn header_shape(layers: u32, hidden: u32, array_len: usize) -> GgufFile {
        use crate::backends::gguf::{types::MetaValue, writer::GgufWriter};
        use crate::quantize::ggml_quants::GgmlType;
        let file = tempfile::NamedTempFile::new().unwrap();
        let kv = [
            ("deepseek4.block_count", MetaValue::U32(layers)),
            ("deepseek4.embedding_length", MetaValue::U32(hidden)),
            (
                "deepseek4.attention.compress_ratios",
                MetaValue::ArrayU32(vec![0; array_len]),
            ),
            (
                "deepseek4.swiglu_clamp_exp",
                MetaValue::ArrayF32(vec![10.0; array_len]),
            ),
            (
                "deepseek4.swiglu_clamp_shexp",
                MetaValue::ArrayF32(vec![10.0; array_len]),
            ),
        ];
        let mut writer = GgufWriter::new(file.reopen().unwrap());
        writer.write_header(1, kv.len() as u64).unwrap();
        for (key, value) in kv {
            writer.write_metadata_kv(key, &value).unwrap();
        }
        let tensor = writer
            .reserve_tensor_info("unused", &[1], GgmlType::F32)
            .unwrap();
        writer.pad_to_alignment().unwrap();
        writer
            .stream_tensor_payload(tensor, &0f32.to_le_bytes())
            .unwrap();
        writer.finalize().unwrap();
        GgufFile::from_file_with_limits(file.reopen().unwrap(), limits()).unwrap()
    }
    #[test]
    fn actual_inspection_rejects_config_expansion_before_missing_config_fallback() {
        for (layers, hidden, count) in [(65, 4096, 65), (1, u32::MAX, 1), (1, 4096, 2)] {
            assert!(inspect(&header_shape(layers, hidden, count)).is_err());
        }
        assert!(matches!(
            inspect(&header_shape(1, 4096, 1)).unwrap(),
            Facts::Unknown { .. }
        ));
    }
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

/// ByteLevel maps each character to one byte or keeps the original UTF8 bytes;
/// from_utf8_lossy expands at most3x. Empty vocab entries normalize to this
/// bounded ASCII family in the actual tokenizer. No token/String clone here.
pub(super) fn decoded_token_bound(g:&GgufFile)->Result<u64>{
 let mut max="[EMPTY_262143]".len() as u64;
 for v in array(g,"tokenizer.ggml.tokens")? {let MetadataValue::String(s)=v else{anyhow::bail!("Q5 token is not UTF8 string")};max=max.max(s.len() as u64);}
 max.checked_mul(3).context("Q5 decoded token byte overflow")
}

#[cfg(test)]mod decoded_bound_controls{
 use super::*;
 #[test]fn actual_borrowed_token_byte_bound_handles_empty_and_long_tokens(){
  use crate::backends::gguf::{writer::GgufWriter,types::MetaValue};
  let f=tempfile::NamedTempFile::new().unwrap();let mut w=GgufWriter::new(f.reopen().unwrap());
  w.write_header(0,1).unwrap();w.write_metadata_kv("tokenizer.ggml.tokens",&MetaValue::ArrayString(vec![String::new(),"x".repeat(100)])).unwrap();w.pad_to_alignment().unwrap();w.finalize().unwrap();
  let g=GgufFile::from_file_with_limits(f.reopen().unwrap(),limits()).unwrap();assert_eq!(decoded_token_bound(&g).unwrap(),300);
 }
}
