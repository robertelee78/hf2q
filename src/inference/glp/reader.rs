//! GLP GGUF reader with spec-conformance gates (spec/GLP.md "Reader
//! conformance" clauses 1–6).
//!
//! Parsing discipline follows `input/hf_download/gguf_probe.rs`: bounded
//! counts, explicit resource limits, no panics on malformed input.
//!
//! Gates enforced here (fatal = hard error, never a fallback):
//! 1. `glp.mode` absent → `add`; present must be `add` or `project`.
//! 2. `glp.hook_point` must be `residual_stream_post_layer`.
//! 3. `glp.spec_version` must be 1.
//! 4. `direction.0` is invalid; `direction.N` applies at layer N.
//! 5. All direction tensors: fp32, 1-D, identical width.
//! 6. `project` vectors never merge with any other control vector
//!    (enforced by the loader API: one vector per engine binding).

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const MAX_METADATA: u64 = 4096;
const MAX_TENSORS: u64 = 4096;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 2_000_000;
const MAX_ALIGNMENT: u64 = 1024 * 1024;
const MAX_DIRECTION_BYTES: u64 = 64 * 1024 * 1024; // any plausible vector stack

const HOOK_POINT_REQUIRED: &str = "residual_stream_post_layer";
// The weightless 2026-09-04 correction: on DeepSeek-V4 the post-layer
// residual is folded by the next layer's fused kernel, so the measured
// hook is the FFN's pre-fold write. In hf2q's native graph the fold is
// materialized by `dispatch_hc_post` before the next layer, so the two
// labels name the same buffer. Accept both; the apply point is unchanged.
const HOOK_POINT_ALIAS: &str = "ffn_out_pre_residual";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlpMode {
    Add,
    Project,
}

#[derive(Debug)]
pub enum GlpError {
    Io(std::io::Error),
    Malformed(String),
    Conformance(String),
}

impl fmt::Display for GlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GLP read error: {e}"),
            Self::Malformed(m) => write!(f, "malformed GLP GGUF: {m}"),
            Self::Conformance(m) => write!(f, "GLP conformance failure: {m}"),
        }
    }
}

impl std::error::Error for GlpError {}
impl From<std::io::Error> for GlpError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A loaded, conformance-checked GLP vector.
#[derive(Debug, Clone)]
pub struct GlpVector {
    pub mode: GlpMode,
    pub alpha_default: f32,
    pub rank: u32,
    /// layer N (1-based per spec) → direction tensor (fp32, width = hidden).
    pub layers: BTreeMap<u32, Vec<f32>>,
    pub width: usize,
    pub content_sha256: Option<String>,
    pub method: Option<String>,
    pub base_model_name: Option<String>,
}

enum MetaValue {
    U32(u32),
    F32(f32),
    #[allow(dead_code)]
    Bool(bool),
    Str(String),
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], GlpError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GlpError::Malformed("offset overflow".into()))?;
        if end > self.bytes.len() {
            return Err(GlpError::Malformed("unexpected end of file".into()));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, GlpError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GlpError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, GlpError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, GlpError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, GlpError> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(GlpError::Malformed(format!(
                "string length {len} exceeds budget"
            )));
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| GlpError::Malformed("invalid UTF-8 in string".into()))
    }
}

/// Metadata value types per GGUF v3.
fn read_meta_value(reader: &mut Reader, value_type: u32) -> Result<MetaValue, GlpError> {
    match value_type {
        4 => Ok(MetaValue::U32(reader.u32()?)),
        6 => Ok(MetaValue::F32(reader.f32()?)),
        7 => Ok(MetaValue::Bool(reader.take(1)?[0] != 0)),
        8 => Ok(MetaValue::Str(reader.string()?)),
        10 => Ok(MetaValue::U32(reader.u64()? as u32)), // u64 → store truncated
        5 => Ok(MetaValue::U32(reader.i32()? as u32)),  // i32
        _ => Err(GlpError::Malformed(format!(
            "unsupported metadata value type {value_type}"
        ))),
    }
}

fn read_string_or_skip_array(reader: &mut Reader, value_type: u32) -> Result<(), GlpError> {
    if value_type == 9 {
        // array: element type + count + elements
        let elem_type = reader.u32()?;
        let count = reader.u64()?;
        if count > MAX_ARRAY_ELEMENTS {
            return Err(GlpError::Malformed(format!(
                "array element count {count} exceeds budget"
            )));
        }
        for _ in 0..count {
            read_meta_value(reader, elem_type)?;
        }
        Ok(())
    } else {
        read_meta_value(reader, value_type).map(|_| ())
    }
}

struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

fn parse_gguf(bytes: &[u8]) -> Result<(BTreeMap<String, MetaValue>, Vec<TensorInfo>, u64), GlpError> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != GGUF_MAGIC {
        return Err(GlpError::Malformed("bad GGUF magic".into()));
    }
    if reader.u32()? != GGUF_VERSION {
        return Err(GlpError::Malformed("GLP requires GGUF v3".into()));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count == 0 || tensor_count > MAX_TENSORS {
        return Err(GlpError::Malformed(format!(
            "tensor count {tensor_count} outside 1..={MAX_TENSORS}"
        )));
    }
    if metadata_count == 0 || metadata_count > MAX_METADATA {
        return Err(GlpError::Malformed(format!(
            "metadata count {metadata_count} outside 1..={MAX_METADATA}"
        )));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        if value_type == 9 {
            read_string_or_skip_array(&mut reader, value_type)?;
            continue;
        }
        let value = read_meta_value(&mut reader, value_type)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let n_dims = reader.u32()?;
        if n_dims == 0 || n_dims > 4 {
            return Err(GlpError::Malformed(format!(
                "tensor {name}: dim count {n_dims} outside 1..=4"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(reader.u64()?);
        }
        let ggml_type = reader.u32()?;
        let offset = reader.u64()?;
        tensors.push(TensorInfo { name, n_dims, dims, ggml_type, offset });
    }

    // tensor data starts at next alignment boundary
    let alignment = match metadata.get("general.alignment") {
        Some(MetaValue::U32(a)) => *a as u64,
        _ => 32,
    };
    if alignment == 0 || alignment > MAX_ALIGNMENT {
        return Err(GlpError::Malformed(format!(
            "alignment {alignment} outside 1..={MAX_ALIGNMENT}"
        )));
    }
    let data_offset = reader.pos.div_ceil(alignment as usize) * alignment as usize;
    Ok((metadata, tensors, data_offset as u64))
}

impl GlpVector {
    /// Load and conformance-check a GLP vector from a GGUF file.
    pub fn load(path: &Path) -> Result<Self, GlpError> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GlpError> {
        let (metadata, tensors, data_offset) = parse_gguf(bytes)?;

        // Gate 1: mode
        let mode = match metadata.get("glp.mode") {
            None => GlpMode::Add,
            Some(MetaValue::Str(s)) if s == "add" => GlpMode::Add,
            Some(MetaValue::Str(s)) if s == "project" => GlpMode::Project,
            Some(MetaValue::Str(s)) => {
                return Err(GlpError::Conformance(format!(
                    "glp.mode {s:?} is not implemented (fatal: never fall back to add)"
                )));
            }
            Some(_) => return Err(GlpError::Malformed("glp.mode must be a string".into())),
        };

        // Gate 2: hook point
        match metadata.get("glp.hook_point") {
            Some(MetaValue::Str(s)) if s == HOOK_POINT_REQUIRED => {}
            Some(MetaValue::Str(s)) if s == HOOK_POINT_ALIAS => {}
            Some(MetaValue::Str(s)) => {
                return Err(GlpError::Conformance(format!(
                    "glp.hook_point {s:?} names a hook this reader does not apply at"
                )));
            }
            Some(_) => return Err(GlpError::Malformed("glp.hook_point must be a string".into())),
            None => {
                return Err(GlpError::Conformance(
                    "glp.hook_point missing; refusing to guess the apply point".into(),
                ));
            }
        }

        // Gate 3: spec version
        match metadata.get("glp.spec_version") {
            Some(MetaValue::U32(1)) => {}
            Some(MetaValue::U32(v)) => {
                return Err(GlpError::Conformance(format!(
                    "glp.spec_version {v} unsupported (this reader implements 1)"
                )));
            }
            Some(_) => return Err(GlpError::Malformed("glp.spec_version must be uint32".into())),
            None => {
                return Err(GlpError::Conformance(
                    "glp.spec_version missing; cannot tell which contract this file holds".into(),
                ));
            }
        }

        let alpha_default = match metadata.get("glp.alpha_default") {
            Some(MetaValue::F32(a)) => *a,
            _ => 1.0,
        };
        let rank = match metadata.get("glp.rank") {
            Some(MetaValue::U32(r)) => *r,
            _ => 1,
        };
        if rank != 1 {
            return Err(GlpError::Conformance(format!(
                "glp.rank {rank} unsupported (rank-1 vectors only at this layer)"
            )));
        }

        let content_sha256 = match metadata.get("glp.content_sha256") {
            Some(MetaValue::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let method = match metadata.get("glp.method") {
            Some(MetaValue::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let base_model_name = match metadata.get("general.base_model.0.name") {
            Some(MetaValue::Str(s)) => Some(s.clone()),
            _ => None,
        };

        // Gate 4: direction tensors
        let mut layers: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        let mut width: Option<usize> = None;
        for tensor in &tensors {
            let Some(suffix) = tensor.name.strip_prefix("direction.") else {
                continue;
            };
            let layer: u32 = suffix.parse().map_err(|_| {
                GlpError::Malformed(format!("direction tensor name {:?} has non-numeric suffix", tensor.name))
            })?;
            if layer == 0 {
                return Err(GlpError::Conformance(
                    "direction.0 is invalid; layers are 1-based".into(),
                ));
            }
            if tensor.n_dims != 1 {
                return Err(GlpError::Conformance(format!(
                    "direction.{layer} must be 1-D (got {} dims)",
                    tensor.n_dims
                )));
            }
            // GGML type 0 = F32
            if tensor.ggml_type != 0 {
                return Err(GlpError::Conformance(format!(
                    "direction.{layer} must be fp32 (ggml type {})",
                    tensor.ggml_type
                )));
            }
            let w = tensor.dims[0] as usize;
            match width {
                None => width = Some(w),
                Some(existing) if existing != w => {
                    return Err(GlpError::Conformance(format!(
                        "direction.{layer} width {w} differs from earlier width {existing}"
                    )));
                }
                _ => {}
            }
            let byte_len = (w as u64) * 4;
            if byte_len > MAX_DIRECTION_BYTES {
                return Err(GlpError::Malformed(format!(
                    "direction.{layer} byte length {byte_len} exceeds budget"
                )));
            }
            let start = data_offset as usize + tensor.offset as usize;
            let end = start + byte_len as usize;
            if end > bytes.len() {
                return Err(GlpError::Malformed(format!(
                    "direction.{layer} data range exceeds file size"
                )));
            }
            let mut values = Vec::with_capacity(w);
            for chunk in bytes[start..end].chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            if layers.insert(layer, values).is_some() {
                return Err(GlpError::Malformed(format!(
                    "duplicate direction.{layer}"
                )));
            }
        }
        if layers.is_empty() {
            return Err(GlpError::Conformance(
                "no direction.N tensors found; not a GLP vector".into(),
            ));
        }

        Ok(Self {
            mode,
            alpha_default,
            rank,
            layers,
            width: width.unwrap_or(0),
            content_sha256,
            method,
            base_model_name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal GGUF v3 writer for fixtures.
    fn build_gguf(
        meta: &[(&str, MetaValue)],
        tensors: &[(&str, Vec<f32>)],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.write_all(b"GGUF").unwrap();
        out.write_all(&3u32.to_le_bytes()).unwrap();
        out.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
        out.write_all(&(meta.len() as u64).to_le_bytes()).unwrap();
        for (key, value) in meta {
            out.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
            out.write_all(key.as_bytes()).unwrap();
            match value {
                MetaValue::U32(v) => {
                    out.write_all(&4u32.to_le_bytes()).unwrap();
                    out.write_all(&v.to_le_bytes()).unwrap();
                }
                MetaValue::F32(v) => {
                    out.write_all(&6u32.to_le_bytes()).unwrap();
                    out.write_all(&v.to_le_bytes()).unwrap();
                }
                MetaValue::Bool(v) => {
                    out.write_all(&7u32.to_le_bytes()).unwrap();
                    out.write_all(&[*v as u8]).unwrap();
                }
                MetaValue::Str(s) => {
                    out.write_all(&8u32.to_le_bytes()).unwrap();
                    out.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                    out.write_all(s.as_bytes()).unwrap();
                }
            }
        }
        // tensor infos
        let mut offset = 0u64;
        for (name, values) in tensors {
            out.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            out.write_all(name.as_bytes()).unwrap();
            out.write_all(&1u32.to_le_bytes()).unwrap(); // n_dims
            out.write_all(&(values.len() as u64).to_le_bytes()).unwrap();
            out.write_all(&0u32.to_le_bytes()).unwrap(); // F32
            out.write_all(&offset.to_le_bytes()).unwrap();
            offset += (values.len() * 4) as u64;
        }
        // align to 32
        let pad = (32 - out.len() % 32) % 32;
        out.extend(std::iter::repeat(0u8).take(pad));
        for (_, values) in tensors {
            for v in values {
                out.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        out
    }

    fn base_meta() -> Vec<(&'static str, MetaValue)> {
        vec![
            ("glp.mode", MetaValue::Str("project".into())),
            ("glp.spec_version", MetaValue::U32(1)),
            ("glp.hook_point", MetaValue::Str("residual_stream_post_layer".into())),
            ("glp.alpha_default", MetaValue::F32(4.0)),
        ]
    }

    #[test]
    fn loads_a_valid_projective_vector() {
        let bytes = build_gguf(
            &base_meta(),
            &[("direction.10", vec![0.1, 0.2, 0.3]), ("direction.11", vec![0.4, 0.5, 0.6])],
        );
        let vector = GlpVector::from_bytes(&bytes).unwrap();
        assert_eq!(vector.mode, GlpMode::Project);
        assert_eq!(vector.alpha_default, 4.0);
        assert_eq!(vector.width, 3);
        assert_eq!(vector.layers.len(), 2);
        assert!(vector.layers.contains_key(&10));
        assert!((vector.layers[&11][2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn absent_mode_defaults_to_add() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "glp.mode");
        let bytes = build_gguf(&meta, &[("direction.3", vec![1.0])]);
        let vector = GlpVector::from_bytes(&bytes).unwrap();
        assert_eq!(vector.mode, GlpMode::Add);
    }

    #[test]
    fn unknown_mode_is_fatal_never_falls_back() {
        let mut meta = base_meta();
        meta[0] = ("glp.mode", MetaValue::Str("ablate".into()));
        let bytes = build_gguf(&meta, &[("direction.3", vec![1.0])]);
        let err = GlpVector::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, GlpError::Conformance(_)));
    }

    #[test]
    fn wrong_hook_point_is_fatal() {
        let mut meta = base_meta();
        meta[2] = ("glp.hook_point", MetaValue::Str("attn.wo_b".into()));
        let bytes = build_gguf(&meta, &[("direction.3", vec![1.0])]);
        assert!(matches!(
            GlpVector::from_bytes(&bytes),
            Err(GlpError::Conformance(_))
        ));
    }

    #[test]
    fn direction_zero_is_invalid() {
        let bytes = build_gguf(&base_meta(), &[("direction.0", vec![1.0])]);
        assert!(matches!(
            GlpVector::from_bytes(&bytes),
            Err(GlpError::Conformance(_))
        ));
    }

    #[test]
    fn missing_spec_version_is_fatal() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "glp.spec_version");
        let bytes = build_gguf(&meta, &[("direction.3", vec![1.0])]);
        assert!(matches!(
            GlpVector::from_bytes(&bytes),
            Err(GlpError::Conformance(_))
        ));
    }

    #[test]
    fn width_mismatch_across_layers_is_fatal() {
        let bytes = build_gguf(
            &base_meta(),
            &[("direction.3", vec![1.0, 2.0]), ("direction.4", vec![1.0])],
        );
        assert!(matches!(
            GlpVector::from_bytes(&bytes),
            Err(GlpError::Conformance(_))
        ));
    }

    #[test]
    fn no_direction_tensors_is_fatal() {
        let bytes = build_gguf(&base_meta(), &[("token_embd.weight", vec![1.0])]);
        assert!(matches!(
            GlpVector::from_bytes(&bytes),
            Err(GlpError::Conformance(_))
        ));
    }
}
