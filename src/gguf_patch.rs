//! Metadata-only GGUF patching utilities.
//!
//! This module intentionally re-emits an existing GGUF from its parsed header
//! and raw tensor payloads. It does not route through `GgufBackend::write`
//! because that writer starts from `QuantizedModel` and may rename, synthesize,
//! or repack tensors. The contract here is narrower: preserve every original
//! tensor byte and append one metadata KV when the file has a known arch.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use mlx_native::gguf::GgufFile;
use tracing::{info, warn};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const GGUF_ALIGNMENT_KEY: &str = "general.alignment";
const KEY_ARCH: &str = "general.architecture";
const KEY_CHAT_TEMPLATE: &str = "tokenizer.chat_template";

const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_I16: u32 = 17;
const GGML_TYPE_IQ4_NL: u32 = 20;

#[derive(Debug)]
pub struct GgufPatchOptions {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub in_place: bool,
    pub dry_run: bool,
}

#[derive(Debug)]
pub enum PatchOutcome {
    Patched {
        output: PathBuf,
        arch: String,
        template_len: usize,
        tensors: usize,
    },
    DryRunWouldPatch {
        arch: String,
        template_len: usize,
        tensors: usize,
    },
    AlreadyHasChatTemplate {
        arch: Option<String>,
    },
    UnknownArch {
        arch: String,
    },
    MissingArch,
}

#[derive(Debug, Clone)]
struct MetadataKv {
    key: String,
    value: MetadataValue,
}

#[derive(Debug, Clone)]
enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        elem_type: u32,
        values: Vec<MetadataValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
struct TensorInfoRaw {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    old_offset: u64,
    new_offset: u64,
    byte_len: u64,
}

#[derive(Debug)]
struct ParsedGguf {
    metadata: Vec<MetadataKv>,
    tensors: Vec<TensorInfoRaw>,
    tensor_data_offset: u64,
    alignment: u64,
}

pub fn patch_chat_template_from_arch(opts: GgufPatchOptions) -> Result<PatchOutcome> {
    let reader_view = GgufFile::open(&opts.input)
        .map_err(|e| anyhow!("failed to open GGUF via mlx_native reader: {e}"))?;
    let arch_from_reader = reader_view.metadata_string(KEY_ARCH).map(str::to_owned);
    let already_from_reader = reader_view.metadata_string(KEY_CHAT_TEMPLATE).is_some();

    let mut parsed = parse_gguf(&opts.input)?;
    let arch = arch_from_reader.or_else(|| metadata_string(&parsed.metadata, KEY_ARCH));

    if already_from_reader || metadata_string(&parsed.metadata, KEY_CHAT_TEMPLATE).is_some() {
        info!(
            arch = arch.as_deref().unwrap_or("<missing>"),
            input = %opts.input.display(),
            "GGUF patch: tokenizer.chat_template already present; skipping"
        );
        return Ok(PatchOutcome::AlreadyHasChatTemplate { arch });
    }

    let Some(arch) = arch else {
        warn!(
            input = %opts.input.display(),
            "GGUF patch: missing general.architecture; skipping chat_template injection"
        );
        return Ok(PatchOutcome::MissingArch);
    };

    let Some(template) = crate::backends::chat_templates::arch_default_chat_template(&arch) else {
        warn!(
            arch = %arch,
            input = %opts.input.display(),
            "GGUF patch: no arch-default chat_template; skipping"
        );
        return Ok(PatchOutcome::UnknownArch { arch });
    };

    if opts.dry_run {
        info!(
            arch = %arch,
            input = %opts.input.display(),
            template_len = template.len(),
            tensors = parsed.tensors.len(),
            "GGUF patch dry-run: would inject tokenizer.chat_template"
        );
        return Ok(PatchOutcome::DryRunWouldPatch {
            arch,
            template_len: template.len(),
            tensors: parsed.tensors.len(),
        });
    }

    parsed.metadata.push(MetadataKv {
        key: KEY_CHAT_TEMPLATE.to_string(),
        value: MetadataValue::String(template.to_string()),
    });

    let output = resolve_output_path(&opts)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
    }

    let write_path = if opts.in_place {
        let file_name = output
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("input.gguf");
        output.with_file_name(format!(".{file_name}.hf2q-gguf-patch.tmp"))
    } else {
        output.clone()
    };

    write_patched_gguf(&opts.input, &write_path, &mut parsed)
        .with_context(|| format!("write patched GGUF {}", write_path.display()))?;

    if opts.in_place {
        fs::rename(&write_path, &output).with_context(|| {
            format!(
                "replace {} with patched temporary {}",
                output.display(),
                write_path.display()
            )
        })?;
    }

    info!(
        arch = %arch,
        input = %opts.input.display(),
        output = %output.display(),
        template_len = template.len(),
        tensors = parsed.tensors.len(),
        "GGUF patch: injected tokenizer.chat_template"
    );

    Ok(PatchOutcome::Patched {
        output,
        arch,
        template_len: template.len(),
        tensors: parsed.tensors.len(),
    })
}

fn resolve_output_path(opts: &GgufPatchOptions) -> Result<PathBuf> {
    if opts.in_place {
        Ok(opts.input.clone())
    } else {
        opts.output
            .clone()
            .ok_or_else(|| anyhow!("missing --output for non-in-place patch"))
    }
}

fn metadata_string(metadata: &[MetadataKv], key: &str) -> Option<String> {
    metadata.iter().find_map(|kv| {
        if kv.key == key {
            match &kv.value {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        }
    })
}

fn parse_gguf(path: &Path) -> Result<ParsedGguf> {
    let file = File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
    let mut r = BufReader::new(file);

    let magic = read_u32(&mut r)?;
    if magic != GGUF_MAGIC {
        bail!("bad GGUF magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}");
    }
    let version = read_u32(&mut r)?;
    if version != GGUF_VERSION {
        bail!("unsupported GGUF version {version}; only v3 is supported");
    }

    let tensor_count = read_u64(&mut r)? as usize;
    let metadata_count = read_u64(&mut r)? as usize;
    let mut metadata = Vec::with_capacity(metadata_count);
    for _ in 0..metadata_count {
        let key = read_gguf_string(&mut r)?;
        let value_type = read_u32(&mut r)?;
        let value = read_metadata_value(&mut r, value_type)?;
        metadata.push(MetadataKv { key, value });
    }

    let alignment = metadata
        .iter()
        .find(|kv| kv.key == GGUF_ALIGNMENT_KEY)
        .and_then(|kv| metadata_as_u64(&kv.value))
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        bail!("invalid GGUF alignment {alignment}; expected a non-zero power of two");
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut r)?;
        let n_dims = read_u32(&mut r)? as usize;
        if n_dims > 8 {
            bail!("tensor '{name}' has {n_dims} dimensions; max supported is 8");
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut r)?);
        }
        let ggml_type = read_u32(&mut r)?;
        let old_offset = read_u64(&mut r)?;
        let byte_len = tensor_byte_len(&dims, ggml_type)
            .with_context(|| format!("tensor '{name}' byte length"))?;
        tensors.push(TensorInfoRaw {
            name,
            dims,
            ggml_type,
            old_offset,
            new_offset: 0,
            byte_len,
        });
    }

    let tensor_info_end = r.stream_position().context("read tensor info position")?;
    let tensor_data_offset = align_up(tensor_info_end, alignment);

    Ok(ParsedGguf {
        metadata,
        tensors,
        tensor_data_offset,
        alignment,
    })
}

fn metadata_as_u64(value: &MetadataValue) -> Option<u64> {
    match value {
        MetadataValue::Uint8(v) => Some(*v as u64),
        MetadataValue::Uint16(v) => Some(*v as u64),
        MetadataValue::Uint32(v) => Some(*v as u64),
        MetadataValue::Uint64(v) => Some(*v),
        MetadataValue::Int8(v) if *v >= 0 => Some(*v as u64),
        MetadataValue::Int16(v) if *v >= 0 => Some(*v as u64),
        MetadataValue::Int32(v) if *v >= 0 => Some(*v as u64),
        MetadataValue::Int64(v) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}

fn write_patched_gguf(input: &Path, output: &Path, parsed: &mut ParsedGguf) -> Result<()> {
    let input_file =
        File::open(input).with_context(|| format!("open input {}", input.display()))?;
    let mut input_reader = BufReader::new(input_file);
    let output_file =
        File::create(output).with_context(|| format!("create output {}", output.display()))?;
    let mut w = BufWriter::new(output_file);

    w.write_all(&GGUF_MAGIC.to_le_bytes())?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;
    w.write_all(&(parsed.tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(parsed.metadata.len() as u64).to_le_bytes())?;

    for kv in &parsed.metadata {
        write_gguf_string(&mut w, &kv.key)?;
        write_metadata_value(&mut w, &kv.value)?;
    }

    let mut next_offset = 0u64;
    for tensor in &mut parsed.tensors {
        next_offset = align_up(next_offset, parsed.alignment);
        tensor.new_offset = next_offset;
        next_offset = next_offset
            .checked_add(tensor.byte_len)
            .ok_or_else(|| anyhow!("tensor data offsets overflow u64"))?;

        write_gguf_string(&mut w, &tensor.name)?;
        w.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
        for dim in &tensor.dims {
            w.write_all(&dim.to_le_bytes())?;
        }
        w.write_all(&tensor.ggml_type.to_le_bytes())?;
        w.write_all(&tensor.new_offset.to_le_bytes())?;
    }

    let header_end = w.stream_position()?;
    let new_data_start = align_up(header_end, parsed.alignment);
    write_zero_padding(&mut w, new_data_start - header_end)?;

    let mut buffer = vec![0u8; 1024 * 1024];
    for tensor in &parsed.tensors {
        let current = w.stream_position()?;
        let target = new_data_start + tensor.new_offset;
        if current > target {
            bail!(
                "writer passed tensor '{}' target offset: current={}, target={}",
                tensor.name,
                current,
                target
            );
        }
        write_zero_padding(&mut w, target - current)?;

        let old_abs = parsed.tensor_data_offset + tensor.old_offset;
        input_reader
            .seek(SeekFrom::Start(old_abs))
            .with_context(|| format!("seek input tensor '{}'", tensor.name))?;
        copy_exact_bytes(
            &mut input_reader,
            &mut w,
            tensor.byte_len,
            &mut buffer,
            &tensor.name,
        )?;
    }

    w.flush()?;
    Ok(())
}

fn copy_exact_bytes<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    mut len: u64,
    buffer: &mut [u8],
    tensor_name: &str,
) -> Result<()> {
    while len > 0 {
        let chunk = len.min(buffer.len() as u64) as usize;
        r.read_exact(&mut buffer[..chunk])
            .with_context(|| format!("read tensor '{tensor_name}' bytes"))?;
        w.write_all(&buffer[..chunk])
            .with_context(|| format!("write tensor '{tensor_name}' bytes"))?;
        len -= chunk as u64;
    }
    Ok(())
}

fn write_zero_padding<W: Write>(w: &mut W, len: u64) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    const ZEROS: [u8; 4096] = [0; 4096];
    let mut remaining = len;
    while remaining > 0 {
        let n = remaining.min(ZEROS.len() as u64) as usize;
        w.write_all(&ZEROS[..n])?;
        remaining -= n as u64;
    }
    Ok(())
}

fn read_metadata_value<R: Read>(r: &mut R, value_type: u32) -> Result<MetadataValue> {
    match value_type {
        TYPE_UINT8 => Ok(MetadataValue::Uint8(read_u8(r)?)),
        TYPE_INT8 => Ok(MetadataValue::Int8(read_i8(r)?)),
        TYPE_UINT16 => Ok(MetadataValue::Uint16(read_u16(r)?)),
        TYPE_INT16 => Ok(MetadataValue::Int16(read_i16(r)?)),
        TYPE_UINT32 => Ok(MetadataValue::Uint32(read_u32(r)?)),
        TYPE_INT32 => Ok(MetadataValue::Int32(read_i32(r)?)),
        TYPE_FLOAT32 => Ok(MetadataValue::Float32(read_f32(r)?)),
        TYPE_BOOL => Ok(MetadataValue::Bool(read_u8(r)? != 0)),
        TYPE_STRING => Ok(MetadataValue::String(read_gguf_string(r)?)),
        TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            if elem_type == TYPE_ARRAY {
                bail!("nested GGUF metadata arrays are not supported");
            }
            let len = read_u64(r)? as usize;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_metadata_value(r, elem_type)?);
            }
            Ok(MetadataValue::Array { elem_type, values })
        }
        TYPE_UINT64 => Ok(MetadataValue::Uint64(read_u64(r)?)),
        TYPE_INT64 => Ok(MetadataValue::Int64(read_i64(r)?)),
        TYPE_FLOAT64 => Ok(MetadataValue::Float64(read_f64(r)?)),
        other => bail!("unknown GGUF metadata value type {other}"),
    }
}

fn write_metadata_value<W: Write>(w: &mut W, value: &MetadataValue) -> Result<()> {
    match value {
        MetadataValue::Uint8(v) => {
            w.write_all(&TYPE_UINT8.to_le_bytes())?;
            w.write_all(&[*v])?;
        }
        MetadataValue::Int8(v) => {
            w.write_all(&TYPE_INT8.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
        }
        MetadataValue::Uint16(v) => {
            w.write_all(&TYPE_UINT16.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Int16(v) => {
            w.write_all(&TYPE_INT16.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Uint32(v) => {
            w.write_all(&TYPE_UINT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Int32(v) => {
            w.write_all(&TYPE_INT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Float32(v) => {
            w.write_all(&TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Bool(v) => {
            w.write_all(&TYPE_BOOL.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
        }
        MetadataValue::String(s) => {
            w.write_all(&TYPE_STRING.to_le_bytes())?;
            write_gguf_string(w, s)?;
        }
        MetadataValue::Array { elem_type, values } => {
            w.write_all(&TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&elem_type.to_le_bytes())?;
            w.write_all(&(values.len() as u64).to_le_bytes())?;
            for value in values {
                write_metadata_array_element(w, value)?;
            }
        }
        MetadataValue::Uint64(v) => {
            w.write_all(&TYPE_UINT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Int64(v) => {
            w.write_all(&TYPE_INT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetadataValue::Float64(v) => {
            w.write_all(&TYPE_FLOAT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_metadata_array_element<W: Write>(w: &mut W, value: &MetadataValue) -> Result<()> {
    match value {
        MetadataValue::Uint8(v) => w.write_all(&[*v])?,
        MetadataValue::Int8(v) => w.write_all(&[*v as u8])?,
        MetadataValue::Uint16(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Int16(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Uint32(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Int32(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Float32(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Bool(v) => w.write_all(&[*v as u8])?,
        MetadataValue::String(s) => write_gguf_string(w, s)?,
        MetadataValue::Uint64(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Int64(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Float64(v) => w.write_all(&v.to_le_bytes())?,
        MetadataValue::Array { .. } => bail!("nested GGUF metadata arrays are not supported"),
    }
    Ok(())
}

fn tensor_byte_len(dims: &[u64], ggml_type: u32) -> Result<u64> {
    let total = dims.iter().try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| anyhow!("tensor element count overflow"))
    })?;
    if total == 0 {
        return Ok(0);
    }

    let (block_values, block_bytes) = match ggml_type {
        GGML_TYPE_F32 => (1, 4),
        GGML_TYPE_F16 => (1, 2),
        GGML_TYPE_Q4_0 => (32, 18),
        // ADR-022 Phase 1 — Q5_1 (legacy 5-bit asymmetric, 32-element block).
        GGML_TYPE_Q5_1 => (32, 24),
        GGML_TYPE_Q8_0 => (32, 34),
        GGML_TYPE_Q4_K => (256, 144),
        GGML_TYPE_Q5_K => (256, 176),
        GGML_TYPE_Q6_K => (256, 210),
        GGML_TYPE_I16 => (1, 2),
        // ADR-022 Phase 1 — IQ4_NL (4-bit codebook, 32-element block).
        GGML_TYPE_IQ4_NL => (32, 18),
        other => bail!("unsupported GGML type ID {other}"),
    };
    if total % block_values != 0 {
        bail!("total elements {total} not divisible by GGML block size {block_values}");
    }
    Ok((total / block_values) * block_bytes)
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(i16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::chat_templates::QWEN3_CHATML;

    fn synthetic_gguf(path: &Path, arch: &str, chat_template: Option<&str>) -> Result<()> {
        let mut metadata = vec![
            MetadataKv {
                key: KEY_ARCH.to_string(),
                value: MetadataValue::String(arch.to_string()),
            },
            MetadataKv {
                key: GGUF_ALIGNMENT_KEY.to_string(),
                value: MetadataValue::Uint32(32),
            },
            MetadataKv {
                key: "tokenizer.ggml.tokens".to_string(),
                value: MetadataValue::Array {
                    elem_type: TYPE_STRING,
                    values: vec![
                        MetadataValue::String("<s>".to_string()),
                        MetadataValue::String("</s>".to_string()),
                    ],
                },
            },
        ];
        if let Some(template) = chat_template {
            metadata.push(MetadataKv {
                key: KEY_CHAT_TEMPLATE.to_string(),
                value: MetadataValue::String(template.to_string()),
            });
        }

        let tensors = vec![
            TensorInfoRaw {
                name: "token_embd.weight".to_string(),
                dims: vec![4, 4],
                ggml_type: GGML_TYPE_F32,
                old_offset: 0,
                new_offset: 0,
                byte_len: 64,
            },
            TensorInfoRaw {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![32, 1],
                ggml_type: GGML_TYPE_Q4_0,
                old_offset: 0,
                new_offset: 0,
                byte_len: 18,
            },
        ];
        let data0: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let data1: Vec<u8> = (0..18).map(|i| 200u8 - i as u8).collect();

        let mut parsed = ParsedGguf {
            metadata,
            tensors,
            tensor_data_offset: 0,
            alignment: 32,
        };
        let tmp_input = tempfile::NamedTempFile::new()?;
        write_minimal_gguf(tmp_input.path(), &mut parsed, &[data0, data1])?;
        fs::copy(tmp_input.path(), path)?;
        Ok(())
    }

    fn write_minimal_gguf(
        path: &Path,
        parsed: &mut ParsedGguf,
        payloads: &[Vec<u8>],
    ) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(&GGUF_MAGIC.to_le_bytes())?;
        w.write_all(&GGUF_VERSION.to_le_bytes())?;
        w.write_all(&(parsed.tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(parsed.metadata.len() as u64).to_le_bytes())?;
        for kv in &parsed.metadata {
            write_gguf_string(&mut w, &kv.key)?;
            write_metadata_value(&mut w, &kv.value)?;
        }
        let mut offset = 0u64;
        for tensor in &mut parsed.tensors {
            offset = align_up(offset, parsed.alignment);
            tensor.new_offset = offset;
            write_gguf_string(&mut w, &tensor.name)?;
            w.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
            for dim in &tensor.dims {
                w.write_all(&dim.to_le_bytes())?;
            }
            w.write_all(&tensor.ggml_type.to_le_bytes())?;
            w.write_all(&tensor.new_offset.to_le_bytes())?;
            offset += tensor.byte_len;
        }
        let header_end = w.stream_position()?;
        let data_start = align_up(header_end, parsed.alignment);
        write_zero_padding(&mut w, data_start - header_end)?;
        for (tensor, payload) in parsed.tensors.iter().zip(payloads) {
            let target = data_start + tensor.new_offset;
            let current = w.stream_position()?;
            write_zero_padding(&mut w, target - current)?;
            w.write_all(payload)?;
        }
        w.flush()?;
        Ok(())
    }

    fn raw_tensor_bytes(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
        let parsed = parse_gguf(path)?;
        let mut file = File::open(path)?;
        let mut out = Vec::new();
        for tensor in parsed.tensors {
            file.seek(SeekFrom::Start(
                parsed.tensor_data_offset + tensor.old_offset,
            ))?;
            let mut buf = vec![0u8; tensor.byte_len as usize];
            file.read_exact(&mut buf)?;
            out.push((tensor.name, buf));
        }
        Ok(out)
    }

    #[test]
    fn gguf_patch_injects_qwen35moe_chat_template() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("in.gguf");
        let output = tmp.path().join("out.gguf");
        synthetic_gguf(&input, "qwen35moe", None).unwrap();

        let before_tensors = raw_tensor_bytes(&input).unwrap();
        let outcome = patch_chat_template_from_arch(GgufPatchOptions {
            input: input.clone(),
            output: Some(output.clone()),
            in_place: false,
            dry_run: false,
        })
        .unwrap();

        assert!(matches!(outcome, PatchOutcome::Patched { .. }));
        let gguf = GgufFile::open(&output).unwrap();
        assert_eq!(gguf.metadata_string(KEY_ARCH), Some("qwen35moe"));
        assert_eq!(gguf.metadata_string(KEY_CHAT_TEMPLATE), Some(QWEN3_CHATML));
        assert_eq!(raw_tensor_bytes(&output).unwrap(), before_tensors);
    }

    #[test]
    fn gguf_patch_idempotent_existing_chat_template_leaves_file_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("in.gguf");
        synthetic_gguf(&input, "qwen35moe", Some("custom-template")).unwrap();
        let before = fs::read(&input).unwrap();

        let outcome = patch_chat_template_from_arch(GgufPatchOptions {
            input: input.clone(),
            output: None,
            in_place: true,
            dry_run: false,
        })
        .unwrap();

        assert!(matches!(
            outcome,
            PatchOutcome::AlreadyHasChatTemplate { .. }
        ));
        assert_eq!(fs::read(&input).unwrap(), before);
    }

    #[test]
    fn gguf_patch_unknown_arch_warns_and_skips_gracefully() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("in.gguf");
        let output = tmp.path().join("out.gguf");
        synthetic_gguf(&input, "unknown", None).unwrap();

        let outcome = patch_chat_template_from_arch(GgufPatchOptions {
            input,
            output: Some(output.clone()),
            in_place: false,
            dry_run: false,
        })
        .unwrap();

        assert!(matches!(outcome, PatchOutcome::UnknownArch { .. }));
        assert!(!output.exists());
    }

    #[test]
    fn gguf_patch_dry_run_reports_without_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("in.gguf");
        let output = tmp.path().join("out.gguf");
        synthetic_gguf(&input, "qwen35moe", None).unwrap();

        let outcome = patch_chat_template_from_arch(GgufPatchOptions {
            input,
            output: Some(output.clone()),
            in_place: false,
            dry_run: true,
        })
        .unwrap();

        assert!(matches!(outcome, PatchOutcome::DryRunWouldPatch { .. }));
        assert!(!output.exists());
    }
}
