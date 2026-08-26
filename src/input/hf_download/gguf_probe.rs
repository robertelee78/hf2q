//! Allocation-bounded GGUF header parser for hosted preflight.
//!
//! The general runtime parser intentionally accepts large local GGUF
//! metadata. A network preflight has a stricter trust boundary: every count,
//! string, array, descriptor, and tensor range must fit the authenticated
//! prefix and a small explicit resource budget before payload transfer.

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const MAX_METADATA: u64 = 4096;
const MAX_TENSORS: u64 = 100_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 2_000_000;
const MAX_TOTAL_ARRAY_ELEMENTS: u64 = 4_000_000;
const MAX_ALIGNMENT: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProbedGgufHeader {
    pub(super) architecture: String,
    pub(super) requires_projector: bool,
    pub(super) file_type: u32,
    pub(super) tensor_count: u64,
    pub(super) tensor_data_offset: u64,
    pub(super) token_embedding_type: Option<u32>,
    pub(super) incompatible_tensor: Option<String>,
    pub(super) has_output_norm: bool,
    pub(super) has_block_tensor: bool,
}

pub(super) fn parse_bounded_header(
    bytes: &[u8],
    logical_bytes: u64,
) -> Result<ProbedGgufHeader, String> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != GGUF_MAGIC {
        return Err("bad GGUF magic".into());
    }
    if reader.u32()? != GGUF_VERSION {
        return Err("hosted preflight supports only GGUF v3".into());
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count == 0 || tensor_count > MAX_TENSORS {
        return Err(format!(
            "tensor count {tensor_count} is outside 1..={MAX_TENSORS}"
        ));
    }
    if metadata_count == 0 || metadata_count > MAX_METADATA {
        return Err(format!(
            "metadata count {metadata_count} is outside 1..={MAX_METADATA}"
        ));
    }
    let minimum_tensor_directory = tensor_count
        .checked_mul(24)
        .ok_or("tensor directory size overflow")?;
    if minimum_tensor_directory > logical_bytes {
        return Err("tensor directory cannot fit authenticated object".into());
    }

    let mut architecture = None;
    let mut file_type = None;
    let mut alignment = 32_u64;
    let mut total_array_elements = 0_u64;
    let mut projector_profile = None;
    let mut vision_token_markers = None;
    for _ in 0..metadata_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        match key {
            b"general.architecture" => {
                if architecture.is_some() || value_type != 8 {
                    return Err("general.architecture must be one unique string".into());
                }
                let value = reader.string()?;
                if value.len() > 128 || !value.is_ascii() {
                    return Err("general.architecture is not a bounded ASCII value".into());
                }
                architecture = Some(
                    std::str::from_utf8(value)
                        .map_err(|_| "general.architecture is not UTF-8")?
                        .to_owned(),
                );
            }
            b"general.file_type" => {
                if file_type.is_some() || value_type != 4 {
                    return Err("general.file_type must be one unique u32".into());
                }
                file_type = Some(reader.u32()?);
            }
            b"general.alignment" => {
                if value_type != 4 {
                    return Err("general.alignment must be u32".into());
                }
                alignment = reader.u32()? as u64;
            }
            b"hf2q.vision.projector_profile" => {
                if projector_profile.is_some() || value_type != 8 {
                    return Err("hf2q.vision.projector_profile must be one unique string".into());
                }
                projector_profile = Some(!reader.string()?.is_empty());
            }
            b"tokenizer.ggml.tokens" => {
                if vision_token_markers.is_some() {
                    return Err("tokenizer.ggml.tokens must be unique".into());
                }
                vision_token_markers = Some(reader.string_array_contains_all(
                    value_type,
                    &mut total_array_elements,
                    &[b"<|vision_start|>", b"<|image_pad|>", b"<|vision_end|>"],
                )?);
            }
            _ => reader.skip_value(value_type, &mut total_array_elements)?,
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() || alignment > MAX_ALIGNMENT {
        return Err(format!("invalid GGUF alignment {alignment}"));
    }
    let architecture = architecture.ok_or("GGUF has no general.architecture")?;
    let file_type = file_type.ok_or("GGUF has no general.file_type")?;

    let mut maximum_relative_end = 0_u64;
    let mut token_embedding_type = None;
    let mut incompatible_tensor = None;
    let mut has_output_norm = false;
    let mut has_block_tensor = false;
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dimensions = reader.u32()? as usize;
        if !(1..=8).contains(&dimensions) {
            return Err(format!(
                "tensor dimension count {dimensions} is outside 1..=8"
            ));
        }
        let mut shape = [0_u64; 8];
        for dimension in shape.iter_mut().take(dimensions) {
            *dimension = reader.u64()?;
            if *dimension == 0 {
                return Err("tensor shape contains zero".into());
            }
        }
        let ggml_type = reader.u32()?;
        if name == b"token_embd.weight" {
            if token_embedding_type.replace(ggml_type).is_some() {
                return Err("GGUF has duplicate token_embd.weight tensors".into());
            }
        }
        has_output_norm |= name == b"output_norm.weight";
        has_block_tensor |= name.starts_with(b"blk.0.");
        if incompatible_tensor.is_none() {
            incompatible_tensor = hosted_tensor_incompatibility(&architecture, name, ggml_type);
        }
        let relative_offset = reader.u64()?;
        let tensor_bytes = tensor_bytes(&shape[..dimensions], ggml_type)?;
        let relative_end = relative_offset
            .checked_add(tensor_bytes)
            .ok_or("tensor relative range overflow")?;
        maximum_relative_end = maximum_relative_end.max(relative_end);
    }
    let tensor_data_offset = align_up(reader.position() as u64, alignment)?;
    if tensor_data_offset > bytes.len() as u64 {
        return Err(format!(
            "GGUF directory ends at byte {tensor_data_offset}, beyond authenticated prefix {}",
            bytes.len()
        ));
    }
    let maximum_end = tensor_data_offset
        .checked_add(maximum_relative_end)
        .ok_or("tensor absolute range overflow")?;
    if maximum_end > logical_bytes {
        return Err(format!(
            "tensor directory ends at {maximum_end}, beyond authenticated object {logical_bytes}"
        ));
    }
    Ok(ProbedGgufHeader {
        requires_projector: architecture == "gemma4"
            || architecture.contains("qwen3vl")
            || projector_profile.unwrap_or(false)
            || vision_token_markers.unwrap_or(false),
        architecture,
        file_type,
        tensor_count,
        tensor_data_offset,
        token_embedding_type,
        incompatible_tensor,
        has_output_norm,
        has_block_tensor,
    })
}

fn hosted_tensor_incompatibility(arch: &str, name: &[u8], ggml_type: u32) -> Option<String> {
    use crate::inference::models::qwen35::Qwen35NativeTensorRole;

    let name = std::str::from_utf8(name).ok()?;
    if matches!(arch, "qwen35" | "qwen35moe") {
        let role = Qwen35NativeTensorRole::for_name(name);
        let supported = runtime_ggml_type(ggml_type)
            .and_then(|ggml_type| role.map(|role| role.supports(ggml_type)));
        if matches!(supported, Some(false)) || (role.is_some() && supported.is_none()) {
            return Some(format!(
                "{name} uses unsupported GGML type {ggml_type} for {arch}"
            ));
        }
    }
    None
}

fn runtime_ggml_type(id: u32) -> Option<mlx_native::GgmlType> {
    use mlx_native::GgmlType;

    Some(match id {
        0 => GgmlType::F32,
        1 => GgmlType::F16,
        2 => GgmlType::Q4_0,
        6 => GgmlType::Q5_0,
        7 => GgmlType::Q5_1,
        8 => GgmlType::Q8_0,
        10 => GgmlType::Q2_K,
        11 => GgmlType::Q3_K,
        12 => GgmlType::Q4_K,
        13 => GgmlType::Q5_K,
        14 => GgmlType::Q6_K,
        17 => GgmlType::I16,
        20 => GgmlType::IQ4_NL,
        23 => GgmlType::IQ4_XS,
        26 => GgmlType::I32,
        30 => GgmlType::BF16,
        _ => return None,
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(length)
            .ok_or("GGUF prefix cursor overflow")?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or("authenticated GGUF prefix is incomplete")?;
        self.position = end;
        Ok(value)
    }

    fn skip(&mut self, length: u64) -> Result<(), String> {
        let length = usize::try_from(length).map_err(|_| "GGUF skip exceeds usize")?;
        self.take(length).map(|_| ())
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("exact read"),
        ))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("exact read"),
        ))
    }

    fn string(&mut self) -> Result<&'a [u8], String> {
        let length = self.u64()?;
        if length > MAX_STRING_BYTES {
            return Err(format!(
                "GGUF string length {length} exceeds {MAX_STRING_BYTES} byte hosted cap"
            ));
        }
        let length = usize::try_from(length).map_err(|_| "GGUF string exceeds usize")?;
        self.take(length)
    }

    fn skip_value(
        &mut self,
        value_type: u32,
        total_array_elements: &mut u64,
    ) -> Result<(), String> {
        match value_type {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4 | 5 | 6 => self.skip(4),
            8 => self.string().map(|_| ()),
            9 => {
                let element_type = self.u32()?;
                if element_type == 9 {
                    return Err("nested GGUF metadata arrays are unsupported".into());
                }
                let count = self.u64()?;
                if count > MAX_ARRAY_ELEMENTS {
                    return Err(format!(
                        "GGUF metadata array count {count} exceeds {MAX_ARRAY_ELEMENTS} hosted cap"
                    ));
                }
                *total_array_elements = total_array_elements
                    .checked_add(count)
                    .ok_or("GGUF aggregate array count overflow")?;
                if *total_array_elements > MAX_TOTAL_ARRAY_ELEMENTS {
                    return Err(format!(
                        "GGUF aggregate array elements exceed {MAX_TOTAL_ARRAY_ELEMENTS} hosted cap"
                    ));
                }
                match element_type {
                    0 | 1 | 7 => self.skip(count),
                    2 | 3 => self.skip(count.checked_mul(2).ok_or("array size overflow")?),
                    4 | 5 | 6 => self.skip(count.checked_mul(4).ok_or("array size overflow")?),
                    10 | 11 | 12 => self.skip(count.checked_mul(8).ok_or("array size overflow")?),
                    8 => {
                        for _ in 0..count {
                            self.string()?;
                        }
                        Ok(())
                    }
                    other => Err(format!("unsupported GGUF array element type {other}")),
                }
            }
            10 | 11 | 12 => self.skip(8),
            other => Err(format!("unsupported GGUF metadata value type {other}")),
        }
    }

    fn string_array_contains_all(
        &mut self,
        value_type: u32,
        total_array_elements: &mut u64,
        needles: &[&[u8]],
    ) -> Result<bool, String> {
        if value_type != 9 || self.u32()? != 8 {
            return Err("tokenizer.ggml.tokens must be an array of strings".into());
        }
        let count = self.u64()?;
        if count > MAX_ARRAY_ELEMENTS {
            return Err(format!(
                "GGUF metadata array count {count} exceeds {MAX_ARRAY_ELEMENTS} hosted cap"
            ));
        }
        *total_array_elements = total_array_elements
            .checked_add(count)
            .ok_or("GGUF aggregate array count overflow")?;
        if *total_array_elements > MAX_TOTAL_ARRAY_ELEMENTS {
            return Err(format!(
                "GGUF aggregate array elements exceed {MAX_TOTAL_ARRAY_ELEMENTS} hosted cap"
            ));
        }
        let mut found = vec![false; needles.len()];
        for _ in 0..count {
            let value = self.string()?;
            for (index, needle) in needles.iter().enumerate() {
                found[index] |= value == *needle;
            }
        }
        Ok(found.into_iter().all(|present| present))
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or("GGUF alignment overflow".into())
}

/// GGUF stores dimensions innermost-first. Block-quantized payloads cannot
/// span rows, so the first wire dimension is divided by the block width.
fn tensor_bytes(shape: &[u64], ggml_type: u32) -> Result<u64, String> {
    let ggml_type = runtime_ggml_type(ggml_type)
        .ok_or_else(|| format!("unsupported GGML tensor type {ggml_type}"))?;
    let block_values = u64::from(ggml_type.block_values());
    let block_bytes = u64::from(ggml_type.block_bytes());
    let inner = shape[0];
    if inner % block_values != 0 {
        return Err(format!(
            "innermost dimension {inner} is not divisible by GGML block {block_values}"
        ));
    }
    let outer = shape[1..]
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or("tensor outer shape overflow")?;
    outer
        .checked_mul(inner / block_values)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or("tensor byte length overflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn valid_header(relative_offset: u64) -> (Vec<u8>, u64) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut bytes, "qwen35");
        string(&mut bytes, "general.file_type");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&15_u32.to_le_bytes());
        string(&mut bytes, "token_embd.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&256_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&12_u32.to_le_bytes());
        bytes.extend_from_slice(&relative_offset.to_le_bytes());
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned, 0);
        let logical_bytes = aligned as u64 + relative_offset + 288;
        (bytes, logical_bytes)
    }

    #[test]
    fn valid_supported_header_parses_without_payload_allocation() {
        let (bytes, logical_bytes) = valid_header(0);
        let header = parse_bounded_header(&bytes, logical_bytes).unwrap();
        assert_eq!(header.architecture, "qwen35");
        assert!(!header.requires_projector);
        assert_eq!(header.file_type, 15);
        assert_eq!(header.token_embedding_type, Some(12));
        assert!(header.incompatible_tensor.is_none());
        assert!(!header.has_output_norm);
        assert!(!header.has_block_tensor);
        assert_eq!(header.tensor_data_offset, bytes.len() as u64);
    }

    #[test]
    fn gguf_vision_profile_is_authoritative_for_projector_planning() {
        use crate::backends::gguf::types::MetaValue;
        use crate::backends::gguf::writer::GgufWriter;
        use crate::quantize::ggml_quants::GgmlType;
        use std::io::Cursor;

        let mut writer = GgufWriter::new(Cursor::new(Vec::new()));
        writer.write_header(1, 3).unwrap();
        writer
            .write_metadata_kv("general.architecture", &MetaValue::String("qwen35".into()))
            .unwrap();
        writer
            .write_metadata_kv("general.file_type", &MetaValue::U32(15))
            .unwrap();
        writer
            .write_metadata_kv(
                "hf2q.vision.projector_profile",
                &MetaValue::String("qwen3vl_siglip".into()),
            )
            .unwrap();
        writer
            .reserve_tensor_info("token_embd.weight", &[256, 2], GgmlType::Q4_K)
            .unwrap();
        writer.pad_to_alignment().unwrap();
        writer
            .stream_tensor_payload(0, &vec![0; 2 * GgmlType::Q4_K.row_size(256)])
            .unwrap();
        writer.finalize().unwrap();
        let bytes = writer.into_inner().into_inner();
        let header = parse_bounded_header(&bytes, bytes.len() as u64).unwrap();
        assert!(header.requires_projector);
    }

    #[test]
    fn truncated_prefix_progresses_to_the_same_valid_header() {
        let (bytes, logical_bytes) = valid_header(0);
        for cut in [4, 24, bytes.len() - 1] {
            assert!(parse_bounded_header(&bytes[..cut], logical_bytes).is_err());
        }
        assert!(parse_bounded_header(&bytes, logical_bytes).is_ok());
    }

    #[test]
    fn tensor_range_beyond_authenticated_object_is_rejected() {
        let (bytes, _) = valid_header(1024);
        let error = parse_bounded_header(&bytes, bytes.len() as u64 + 512).unwrap_err();
        assert!(error.contains("beyond authenticated object"), "{error}");
    }

    #[test]
    fn qwen_role_specific_tensor_layout_is_summarized_without_a_descriptor_vector() {
        use crate::inference::models::qwen35::Qwen35NativeTensorRole;

        let types = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 17, 20, 23, 26, 30];
        let roles = [
            ("qwen35", "token_embd.weight"),
            ("qwen35", "blk.0.ffn_gate.weight"),
            ("qwen35moe", "blk.0.ffn_gate_exps.weight"),
            ("qwen35", "blk.0.attn_q.weight"),
            ("qwen35", "output.weight"),
        ];
        for id in types {
            let ggml_type = runtime_ggml_type(id).expect("known runtime GGML type");
            assert_eq!(
                mlx_native::gguf::test_only_ggml_type_from_u32(id).unwrap(),
                ggml_type,
                "bounded-header type map drifted for ID {id}"
            );
            for (arch, name) in roles {
                let expected = Qwen35NativeTensorRole::for_name(name)
                    .expect("classified Qwen runtime role")
                    .supports(ggml_type);
                assert_eq!(
                    hosted_tensor_incompatibility(arch, name.as_bytes(), id).is_none(),
                    expected,
                    "hosted/runtime admission drifted for {name} type {id}"
                );
            }
        }
        assert!(hosted_tensor_incompatibility("qwen35", b"output.weight", 16).is_some());
        assert!(hosted_tensor_incompatibility("qwen35", b"metadata.only", 16).is_none());
    }

    #[test]
    fn malicious_counts_and_lengths_fail_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&(MAX_METADATA + 1).to_le_bytes());
        assert!(parse_bounded_header(&bytes, 1024).is_err());

        bytes[16..24].copy_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&(MAX_STRING_BYTES + 1).to_le_bytes());
        assert!(parse_bounded_header(&bytes, 1024).is_err());
    }
}
