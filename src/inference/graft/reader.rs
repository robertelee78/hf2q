//! Graft GGUF reader with ADR-059 conformance gates.
//!
//! Parsing discipline mirrors `inference/glp/reader.rs` (which follows
//! `input/hf_download/gguf_probe.rs`): bounded counts, explicit resource
//! limits, checked file-controlled arithmetic, no panics on malformed
//! input.
//!
//! Gates enforced here (fatal = hard error, never a fallback):
//! 1. `graft.mode` is REQUIRED and must be `splice_prefix`. Unlike
//!    `glp.mode` (absent → `add`, per the GLP spec), a graft is
//!    cache-integrity state — a missing operation is refused, not
//!    defaulted.
//! 2. `graft.hook_point` must be a recognized splice site. Recognized
//!    but not-yet-implemented staged sites (`window_tail_kv`,
//!    `compressed_kv`, `recurrent_state`) are fatal with a named
//!    message — the reader never silently mis-splices a site it does
//!    not implement (the `glp.mode` pattern applied to sites).
//! 3. `graft.spec_version` must be 1.
//! 4. Tensors are `graft.k.<layer>` / `graft.v.<layer>`, fp32, 3-D
//!    `[n_slots, n_kv_heads, head_dim]`, row-major. Layer indices are
//!    0-based graph layers (unlike GLP's 1-based `direction.N` — a
//!    Qwen3.5 full-attention layer 3 or a Gemma-4 Full layer 0 are
//!    valid graft targets). K and V must cover identical layer sets
//!    with identical shapes; `n_kv_heads` and `head_dim` must be
//!    consistent across every tensor in the file.
//! 5. `graft.n_slots == 0` is the explicit canary artifact: no K/V
//!    tensors are permitted alongside it. `n_slots >= 1` requires at
//!    least one K/V layer pair.
//! 6. `graft.content_sha256`, when declared, is verified over raw F32
//!    bytes in a fixed order: layers ascending by numeric index, K then
//!    V within each layer — excluding metadata and GGUF alignment
//!    padding, before any device placement or normalization.
//! 7. RoPE declaration is REQUIRED (`graft.rope_theta`, f32;
//!    `graft.rotary_dim`, u32): cache K is post-RoPE, so a bank without
//!    a RoPE identity cannot be validated at bind and is refused here.
//!    `graft.position_base` (u32, default 0) and
//!    `graft.mrope_interleaved` (bool, default false) carry the family
//!    flags the bind checks.

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
// A phantom-kv bank for a 4B model is ~18 MB; allow generous headroom for
// larger hidden widths and deeper stacks while staying bounded.
const MAX_GRAFT_TENSOR_BYTES: u64 = 256 * 1024 * 1024;

/// Where a graft bank is spliced (ADR-059 splice-site matrix).
///
/// Sites are different cache media, not synonyms: a bank is derived for
/// one site's tensor geometry and is meaningless at another. The staged
/// sites are recognized names so a future file fails with a precise
/// "not implemented by this reader" instead of "unknown site".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraftHookPoint {
    /// Reserved leading slots of full-attention layers (global,
    /// append-only KV). Ships in v1: Qwen3.5/3.6/3.8 full-attn layers,
    /// Gemma-4 Full layers.
    FullAttnKv,
    /// Sliding-window layers: the graft rides the window tail and is
    /// re-injected as the window slides. Staged site 2 (Gemma-4
    /// Sliding layers) — not implemented; loading is fatal.
    WindowTailKv,
    /// DeepSeek-V4 compressor output region. Staged site 3 — not
    /// implemented; loading is fatal.
    CompressedKv,
    /// DeltaNet conv/recurrent state buffers. Staged site 4 (the site
    /// that lifts Qwen hybrid coverage beyond full-attn layers) — not
    /// implemented; loading is fatal.
    RecurrentState,
}

impl GraftHookPoint {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullAttnKv => "full_attn_kv",
            Self::WindowTailKv => "window_tail_kv",
            Self::CompressedKv => "compressed_kv",
            Self::RecurrentState => "recurrent_state",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "full_attn_kv" => Some(Self::FullAttnKv),
            "window_tail_kv" => Some(Self::WindowTailKv),
            "compressed_kv" => Some(Self::CompressedKv),
            "recurrent_state" => Some(Self::RecurrentState),
            _ => None,
        }
    }

    fn implemented(self) -> bool {
        matches!(self, Self::FullAttnKv)
    }
}

/// How the bank is spliced. v1 implements prefix splicing only
/// (positions `0..N`, phantom-kv's invariant-position contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraftMode {
    SplicePrefix,
}

/// Provenance: how the bank was derived (phantom-kv's three kinds). All
/// three splice identically; the kind records the derivation and is
/// carried for audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraftKind {
    /// K/V extracted from a real cached prefill.
    PrefillKv,
    /// Learned soft-prompt tokens cached as K/V.
    SoftPromptKv,
    /// Learned direct K/V bank (the phantom-kv v3 arm).
    DirectKv,
}

impl GraftKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "prefill_kv" => Some(Self::PrefillKv),
            "softprompt_kv" => Some(Self::SoftPromptKv),
            "direct_kv" => Some(Self::DirectKv),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum GraftError {
    Io(std::io::Error),
    Malformed(String),
    Conformance(String),
}

impl fmt::Display for GraftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "graft read error: {e}"),
            Self::Malformed(m) => write!(f, "malformed graft GGUF: {m}"),
            Self::Conformance(m) => write!(f, "graft conformance failure: {m}"),
        }
    }
}

impl std::error::Error for GraftError {}
impl From<std::io::Error> for GraftError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// One layer's K/V pair, flattened row-major over
/// `[n_slots, n_kv_heads, head_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraftLayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// A loaded, conformance-checked graft bank.
#[derive(Debug, Clone)]
pub struct GraftBank {
    pub mode: GraftMode,
    pub kind: GraftKind,
    /// The declared splice site. The family bind must match it exactly.
    pub hook_point: GraftHookPoint,
    /// Bank length N; the graft occupies cache positions `0..N`.
    /// Zero is the explicit canary artifact (no tensors).
    pub n_slots: u32,
    /// Graph layer → K/V pair. Layer indices are 0-based.
    pub layers: BTreeMap<u32, GraftLayerKv>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// RoPE identity of the bank (cache K is post-RoPE). Required.
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub position_base: u32,
    pub mrope_interleaved: bool,
    pub content_sha256: Option<String>,
    /// Quantization lane the bank was derived against (e.g. `bf16`,
    /// `Q4_K_M`). Informational; cross-lane loads warn at bind.
    pub quant_lane: Option<String>,
}

enum MetaValue {
    U32(u32),
    F32(f32),
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
    fn take(&mut self, n: usize) -> Result<&'a [u8], GraftError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GraftError::Malformed("offset overflow".into()))?;
        if end > self.bytes.len() {
            return Err(GraftError::Malformed("unexpected end of file".into()));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, GraftError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GraftError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, GraftError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, GraftError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, GraftError> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(GraftError::Malformed(format!(
                "string length {len} exceeds budget"
            )));
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| GraftError::Malformed("invalid UTF-8 in string".into()))
    }
}

/// Metadata value types per GGUF v3.
fn read_meta_value(reader: &mut Reader, value_type: u32) -> Result<MetaValue, GraftError> {
    match value_type {
        4 => Ok(MetaValue::U32(reader.u32()?)),
        6 => Ok(MetaValue::F32(reader.f32()?)),
        7 => Ok(MetaValue::Bool(reader.take(1)?[0] != 0)),
        8 => Ok(MetaValue::Str(reader.string()?)),
        10 => Ok(MetaValue::U32(reader.u64()? as u32)), // u64 → store truncated
        5 => Ok(MetaValue::U32(reader.i32()? as u32)),  // i32
        _ => Err(GraftError::Malformed(format!(
            "unsupported metadata value type {value_type}"
        ))),
    }
}

fn read_string_or_skip_array(reader: &mut Reader, value_type: u32) -> Result<(), GraftError> {
    if value_type == 9 {
        // array: element type + count + elements
        let elem_type = reader.u32()?;
        let count = reader.u64()?;
        if count > MAX_ARRAY_ELEMENTS {
            return Err(GraftError::Malformed(format!(
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

fn parse_gguf(
    bytes: &[u8],
) -> Result<(BTreeMap<String, MetaValue>, Vec<TensorInfo>, u64), GraftError> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != GGUF_MAGIC {
        return Err(GraftError::Malformed("bad GGUF magic".into()));
    }
    if reader.u32()? != GGUF_VERSION {
        return Err(GraftError::Malformed("graft requires GGUF v3".into()));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    // Zero tensors is a VALID graft file: the zero-slot canary artifact
    // (graft.n_slots=0) carries no K/V tensors by definition. The
    // graft-specific gates below reject emptiness that matters
    // (n_slots>0 with no tensors). Only the upper bound is fatal here.
    if tensor_count > MAX_TENSORS {
        return Err(GraftError::Malformed(format!(
            "tensor count {tensor_count} outside 0..={MAX_TENSORS}"
        )));
    }
    if metadata_count == 0 || metadata_count > MAX_METADATA {
        return Err(GraftError::Malformed(format!(
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
            return Err(GraftError::Malformed(format!(
                "tensor {name}: dim count {n_dims} outside 1..=4"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(reader.u64()?);
        }
        let ggml_type = reader.u32()?;
        let offset = reader.u64()?;
        tensors.push(TensorInfo {
            name,
            n_dims,
            dims,
            ggml_type,
            offset,
        });
    }

    // tensor data starts at next alignment boundary
    let alignment = match metadata.get("general.alignment") {
        Some(MetaValue::U32(a)) => *a as u64,
        _ => 32,
    };
    if alignment == 0 || alignment > MAX_ALIGNMENT {
        return Err(GraftError::Malformed(format!(
            "alignment {alignment} outside 1..={MAX_ALIGNMENT}"
        )));
    }
    let data_offset = reader.pos.div_ceil(alignment as usize) * alignment as usize;
    Ok((metadata, tensors, data_offset as u64))
}

/// Raw tensor byte range for a parsed tensor, with checked arithmetic
/// and the file-size bound (S11 discipline: all file-controlled
/// arithmetic is checked).
fn tensor_byte_range(
    bytes: &[u8],
    tensor: &TensorInfo,
    data_offset: u64,
) -> Result<(usize, usize), GraftError> {
    let byte_len: u64 = tensor
        .dims
        .iter()
        .try_fold(4u64, |acc, &d| {
            acc.checked_mul(d).filter(|&v| v <= MAX_GRAFT_TENSOR_BYTES)
        })
        .ok_or_else(|| {
            GraftError::Malformed(format!(
                "tensor {:?}: byte length exceeds budget",
                tensor.name
            ))
        })?;
    let tensor_offset = usize::try_from(tensor.offset).map_err(|_| {
        GraftError::Malformed(format!(
            "tensor {:?} offset {} exceeds the address space",
            tensor.name, tensor.offset
        ))
    })?;
    let start = (data_offset as usize)
        .checked_add(tensor_offset)
        .ok_or_else(|| {
            GraftError::Malformed(format!(
                "tensor {:?} data start (offset {tensor_offset}) overflows",
                tensor.name
            ))
        })?;
    let end = start
        .checked_add(byte_len as usize)
        .ok_or_else(|| GraftError::Malformed(format!(
            "tensor {:?} data end (start {start} + {byte_len}) overflows",
            tensor.name
        )))?;
    if end > bytes.len() {
        return Err(GraftError::Malformed(format!(
            "tensor {:?} data range exceeds file size",
            tensor.name
        )));
    }
    Ok((start, end))
}

fn read_f32_tensor(bytes: &[u8], tensor: &TensorInfo, data_offset: u64) -> Result<Vec<f32>, GraftError> {
    let (start, end) = tensor_byte_range(bytes, tensor, data_offset)?;
    let count = (end - start) / 4;
    let mut values = Vec::with_capacity(count);
    for chunk in bytes[start..end].chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

impl GraftBank {
    /// Load and conformance-check a graft bank from a GGUF file.
    pub fn load(path: &Path) -> Result<Self, GraftError> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GraftError> {
        let (metadata, tensors, data_offset) = parse_gguf(bytes)?;

        // Gate 1: mode — REQUIRED (a graft is cache-integrity state; a
        // missing operation is refused, not defaulted — deliberate
        // divergence from glp.mode's absent→add spec default).
        let mode = match metadata.get("graft.mode") {
            Some(MetaValue::Str(s)) if s == "splice_prefix" => GraftMode::SplicePrefix,
            Some(MetaValue::Str(s)) => {
                return Err(GraftError::Conformance(format!(
                    "graft.mode {s:?} is not implemented (fatal: never fall back to splice_prefix)"
                )));
            }
            Some(_) => return Err(GraftError::Malformed("graft.mode must be a string".into())),
            None => {
                return Err(GraftError::Conformance(
                    "graft.mode missing; refusing to guess the splice operation".into(),
                ));
            }
        };

        // Gate 2: splice site — recognized values only; staged sites are
        // fatal with a named message.
        let hook_point = match metadata.get("graft.hook_point") {
            Some(MetaValue::Str(s)) => GraftHookPoint::parse(s).ok_or_else(|| {
                GraftError::Conformance(format!(
                    "graft.hook_point {s:?} is not a recognized splice site \
                     (recognized: full_attn_kv, window_tail_kv, compressed_kv, \
                     recurrent_state)"
                ))
            })?,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.hook_point must be a string".into(),
                ))
            }
            None => {
                return Err(GraftError::Conformance(
                    "graft.hook_point missing; refusing to guess the splice site".into(),
                ));
            }
        };
        if !hook_point.implemented() {
            return Err(GraftError::Conformance(format!(
                "graft.hook_point {:?} is a recognized staged site but is not \
                 implemented by this reader (ADR-059 site matrix); refusing to \
                 mis-splice it as full_attn_kv",
                hook_point.as_str()
            )));
        }

        // Gate 3: spec version.
        match metadata.get("graft.spec_version") {
            Some(MetaValue::U32(1)) => {}
            Some(MetaValue::U32(v)) => {
                return Err(GraftError::Conformance(format!(
                    "graft.spec_version {v} unsupported (this reader implements 1)"
                )));
            }
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.spec_version must be uint32".into(),
                ))
            }
            None => {
                return Err(GraftError::Conformance(
                    "graft.spec_version missing; cannot tell which contract this file holds".into(),
                ));
            }
        }

        // Provenance kind — required, one of phantom-kv's three.
        let kind = match metadata.get("graft.kind") {
            Some(MetaValue::Str(s)) => GraftKind::parse(s).ok_or_else(|| {
                GraftError::Conformance(format!(
                    "graft.kind {s:?} is not a recognized derivation kind \
                     (recognized: prefill_kv, softprompt_kv, direct_kv)"
                ))
            })?,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.kind must be a string".into(),
                ))
            }
            None => {
                return Err(GraftError::Conformance(
                    "graft.kind missing; provenance is required on a cache-integrity artifact".into(),
                ));
            }
        };

        let n_slots = match metadata.get("graft.n_slots") {
            Some(MetaValue::U32(n)) => *n,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.n_slots must be uint32".into(),
                ))
            }
            None => {
                return Err(GraftError::Conformance(
                    "graft.n_slots missing; the bank length defines the splice region".into(),
                ));
            }
        };

        // Gate 7: RoPE declaration — required (cache K is post-RoPE).
        let rope_theta = match metadata.get("graft.rope_theta") {
            Some(MetaValue::F32(t)) => *t,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.rope_theta must be float32".into(),
                ))
            }
            None => {
                return Err(GraftError::Conformance(
                    "graft.rope_theta missing; a post-RoPE K bank cannot be \
                     validated at bind without its RoPE identity".into(),
                ));
            }
        };
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(GraftError::Malformed(format!(
                "graft.rope_theta {rope_theta} must be finite and positive"
            )));
        }
        let rotary_dim = match metadata.get("graft.rotary_dim") {
            Some(MetaValue::U32(d)) => *d,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.rotary_dim must be uint32".into(),
                ))
            }
            None => {
                return Err(GraftError::Conformance(
                    "graft.rotary_dim missing; a post-RoPE K bank cannot be \
                     validated at bind without its RoPE identity".into(),
                ));
            }
        };
        let position_base = match metadata.get("graft.position_base") {
            Some(MetaValue::U32(p)) => *p,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.position_base must be uint32".into(),
                ))
            }
            None => 0,
        };
        let mrope_interleaved = match metadata.get("graft.mrope_interleaved") {
            Some(MetaValue::Bool(b)) => *b,
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.mrope_interleaved must be bool".into(),
                ))
            }
            None => false,
        };

        let content_sha256 = match metadata.get("graft.content_sha256") {
            Some(MetaValue::Str(s))
                if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) =>
            {
                Some(s.to_ascii_lowercase())
            }
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.content_sha256 must be a 64-hex SHA-256 string".into(),
                ))
            }
            None => None,
        };
        // Checkpoint identity is NOT read here: it lives in the
        // GGUF-conventional `general.base_model.0.*` keys and is
        // validated by the compatibility module through the same
        // `CheckpointIdentity` trust boundary GLP uses (ADR-059).
        let quant_lane = match metadata.get("graft.quant_lane") {
            Some(MetaValue::Str(s)) => Some(s.clone()),
            Some(_) => {
                return Err(GraftError::Malformed(
                    "graft.quant_lane must be a string".into(),
                ))
            }
            None => None,
        };

        // Gate 4/5: K/V tensors.
        let mut layers: BTreeMap<u32, (Option<Vec<f32>>, Option<Vec<f32>>)> = BTreeMap::new();
        let mut raw_bytes: BTreeMap<u32, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
        let mut n_kv_heads: Option<usize> = None;
        let mut head_dim: Option<usize> = None;
        for tensor in &tensors {
            let (side, suffix) = match tensor.name.strip_prefix("graft.k.") {
                Some(suffix) => (0, suffix),
                None => match tensor.name.strip_prefix("graft.v.") {
                    Some(suffix) => (1, suffix),
                    None => continue,
                },
            };
            let layer: u32 = suffix.parse().map_err(|_| {
                GraftError::Malformed(format!(
                    "graft tensor name {:?} has non-numeric layer suffix",
                    tensor.name
                ))
            })?;
            let name = tensor.name.as_str();
            if n_slots == 0 {
                return Err(GraftError::Conformance(format!(
                    "graft.n_slots is 0 (canary artifact) but tensor {name:?} \
                     is present; a canary bank carries no K/V tensors"
                )));
            }
            if tensor.n_dims != 3 {
                return Err(GraftError::Conformance(format!(
                    "{name} must be 3-D [n_slots, n_kv_heads, head_dim] \
                     (got {} dims)",
                    tensor.n_dims
                )));
            }
            if tensor.ggml_type != 0 {
                return Err(GraftError::Conformance(format!(
                    "{name} must be fp32 (ggml type {})",
                    tensor.ggml_type
                )));
            }
            if tensor.dims[0] != n_slots as u64 {
                return Err(GraftError::Conformance(format!(
                    "{name} slot dim {} disagrees with graft.n_slots {n_slots}",
                    tensor.dims[0]
                )));
            }
            let heads = usize::try_from(tensor.dims[1]).map_err(|_| {
                GraftError::Malformed(format!(
                    "{name} n_kv_heads {} exceeds the address space",
                    tensor.dims[1]
                ))
            })?;
            let dim = usize::try_from(tensor.dims[2]).map_err(|_| {
                GraftError::Malformed(format!(
                    "{name} head_dim {} exceeds the address space",
                    tensor.dims[2]
                ))
            })?;
            if heads == 0 || dim == 0 {
                return Err(GraftError::Conformance(format!(
                    "{name} has a zero n_kv_heads or head_dim"
                )));
            }
            match n_kv_heads {
                None => n_kv_heads = Some(heads),
                Some(existing) if existing != heads => {
                    return Err(GraftError::Conformance(format!(
                        "{name} n_kv_heads {heads} differs from earlier {existing}"
                    )));
                }
                _ => {}
            }
            match head_dim {
                None => head_dim = Some(dim),
                Some(existing) if existing != dim => {
                    return Err(GraftError::Conformance(format!(
                        "{name} head_dim {dim} differs from earlier {existing}"
                    )));
                }
                _ => {}
            }
            let values = read_f32_tensor(bytes, tensor, data_offset)?;
            let entry = layers.entry(layer).or_insert((None, None));
            let slot = if side == 0 { &mut entry.0 } else { &mut entry.1 };
            if slot.is_some() {
                return Err(GraftError::Malformed(format!("duplicate {name}")));
            }
            *slot = Some(values);
            let (start, end) = tensor_byte_range(bytes, tensor, data_offset)?;
            let raw_entry = raw_bytes.entry(layer).or_insert((Vec::new(), Vec::new()));
            if side == 0 {
                raw_entry.0 = bytes[start..end].to_vec();
            } else {
                raw_entry.1 = bytes[start..end].to_vec();
            }
        }

        if n_slots == 0 {
            // Canary artifact: no tensors permitted (checked above); an
            // otherwise-valid empty bank loads for the no-op canary.
            if !layers.is_empty() {
                return Err(GraftError::Conformance(
                    "canary bank (n_slots=0) must carry no layers".into(),
                ));
            }
        } else {
            if layers.is_empty() {
                return Err(GraftError::Conformance(
                    "no graft.k.<layer>/graft.v.<layer> tensors found; not a graft bank".into(),
                ));
            }
            for (layer, (k, v)) in &layers {
                match (k, v) {
                    (Some(_), Some(_)) => {}
                    (Some(_), None) => {
                        return Err(GraftError::Conformance(format!(
                            "layer {layer} has graft.k but no graft.v"
                        )))
                    }
                    (None, Some(_)) => {
                        return Err(GraftError::Conformance(format!(
                            "layer {layer} has graft.v but no graft.k"
                        )))
                    }
                    (None, None) => unreachable!("entry exists only when a side was stored"),
                }
            }
        }

        // Gate 6: content hash — layers ascending, K then V within each
        // layer, raw F32 bytes, before any normalization or placement.
        if let Some(expected) = content_sha256.as_deref() {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            for (k, v) in raw_bytes.values() {
                hasher.update(k);
                hasher.update(v);
            }
            let actual = hex::encode(hasher.finalize());
            if actual != expected {
                return Err(GraftError::Conformance(format!(
                    "graft.content_sha256 mismatch: declared {expected}, computed {actual}"
                )));
            }
        }

        let layers = layers
            .into_iter()
            .map(|(layer, (k, v))| {
                let k = k.expect("pair completeness checked above");
                let v = v.expect("pair completeness checked above");
                if k.len() != v.len() {
                    return Err(GraftError::Conformance(format!(
                        "layer {layer}: graft.k and graft.v element counts differ \
                         ({} vs {}) — shapes were validated equal, so this is a torn file",
                        k.len(),
                        v.len()
                    )));
                }
                Ok((layer, GraftLayerKv { k, v }))
            })
            .collect::<Result<BTreeMap<u32, GraftLayerKv>, GraftError>>()?;

        Ok(Self {
            mode,
            kind,
            hook_point,
            n_slots,
            layers,
            n_kv_heads: n_kv_heads.unwrap_or(0),
            head_dim: head_dim.unwrap_or(0),
            rope_theta,
            rotary_dim,
            position_base,
            mrope_interleaved,
            content_sha256,
            quant_lane,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal GGUF v3 writer for fixtures. Tensors are
    /// `(name, dims, values)` with `values.len() == product(dims)`.
    fn build_gguf(
        meta: &[(&str, MetaValue)],
        tensors: &[(&str, Vec<u64>, Vec<f32>)],
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
        for (name, dims, values) in tensors {
            out.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            out.write_all(name.as_bytes()).unwrap();
            out.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
            for d in dims {
                out.write_all(&d.to_le_bytes()).unwrap();
            }
            out.write_all(&0u32.to_le_bytes()).unwrap(); // F32
            out.write_all(&offset.to_le_bytes()).unwrap();
            offset += (values.len() * 4) as u64;
        }
        // align to 32
        let pad = (32 - out.len() % 32) % 32;
        out.extend(std::iter::repeat(0u8).take(pad));
        for (_, _, values) in tensors {
            for v in values {
                out.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        out
    }

    fn base_meta() -> Vec<(&'static str, MetaValue)> {
        vec![
            ("graft.mode", MetaValue::Str("splice_prefix".into())),
            ("graft.spec_version", MetaValue::U32(1)),
            ("graft.hook_point", MetaValue::Str("full_attn_kv".into())),
            ("graft.kind", MetaValue::Str("direct_kv".into())),
            ("graft.n_slots", MetaValue::U32(2)),
            ("graft.rope_theta", MetaValue::F32(1e7)),
            ("graft.rotary_dim", MetaValue::U32(64)),
        ]
    }

    /// Two layers (3 and 7 — Qwen3.5 full-attn indices), 2 slots,
    /// 1 KV head, head_dim 2 → 4 elements per tensor.
    fn bank_tensors() -> Vec<(&'static str, Vec<u64>, Vec<f32>)> {
        let dims = vec![2u64, 1, 2];
        vec![
            ("graft.k.3", dims.clone(), vec![1.0, 2.0, 3.0, 4.0]),
            ("graft.v.3", dims.clone(), vec![5.0, 6.0, 7.0, 8.0]),
            ("graft.k.7", dims.clone(), vec![9.0, 10.0, 11.0, 12.0]),
            ("graft.v.7", dims.clone(), vec![13.0, 14.0, 15.0, 16.0]),
        ]
    }

    #[test]
    fn loads_a_valid_splice_prefix_bank() {
        let bytes = build_gguf(&base_meta(), &bank_tensors());
        let bank = GraftBank::from_bytes(&bytes).unwrap();
        assert_eq!(bank.mode, GraftMode::SplicePrefix);
        assert_eq!(bank.kind, GraftKind::DirectKv);
        assert_eq!(bank.hook_point, GraftHookPoint::FullAttnKv);
        assert_eq!(bank.n_slots, 2);
        assert_eq!(bank.layers.len(), 2);
        assert_eq!(bank.n_kv_heads, 1);
        assert_eq!(bank.head_dim, 2);
        assert_eq!(bank.rope_theta, 1e7);
        assert_eq!(bank.rotary_dim, 64);
        assert_eq!(bank.position_base, 0);
        assert!(!bank.mrope_interleaved);
        assert!((bank.layers[&3].k[3] - 4.0).abs() < 1e-6);
        assert!((bank.layers[&7].v[0] - 13.0).abs() < 1e-6);
    }

    #[test]
    fn layer_zero_is_valid_unlike_glp_direction_zero() {
        // Graph layer 0 is a valid graft target (Gemma-4 Full layer 0);
        // GLP's direction.0 ban is a spec rule of that container, not this one.
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.n_slots");
        meta.push(("graft.n_slots", MetaValue::U32(1)));
        let dims = vec![1u64, 1, 1];
        let bytes = build_gguf(
            &meta,
            &[
                ("graft.k.0", dims.clone(), vec![1.0]),
                ("graft.v.0", dims, vec![2.0]),
            ],
        );
        let bank = GraftBank::from_bytes(&bytes).unwrap();
        assert!(bank.layers.contains_key(&0));
    }

    #[test]
    fn missing_mode_is_fatal_refuses_to_guess() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.mode");
        let bytes = build_gguf(&meta, &bank_tensors());
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, GraftError::Conformance(_)));
        assert!(err.to_string().contains("graft.mode missing"));
    }

    #[test]
    fn unknown_mode_is_fatal_never_falls_back() {
        let mut meta = base_meta();
        meta[0] = ("graft.mode", MetaValue::Str("splice_suffix".into()));
        let bytes = build_gguf(&meta, &bank_tensors());
        assert!(matches!(
            GraftBank::from_bytes(&bytes),
            Err(GraftError::Conformance(_))
        ));
    }

    #[test]
    fn unknown_site_is_fatal() {
        let mut meta = base_meta();
        meta[2] = ("graft.hook_point", MetaValue::Str("kv_cache".into()));
        let bytes = build_gguf(&meta, &bank_tensors());
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, GraftError::Conformance(_)));
        assert!(err.to_string().contains("not a recognized splice site"));
    }

    #[test]
    fn staged_sites_are_recognized_but_fatal() {
        for site in ["window_tail_kv", "compressed_kv", "recurrent_state"] {
            let mut meta = base_meta();
            meta[2] = ("graft.hook_point", MetaValue::Str(site.into()));
            let bytes = build_gguf(&meta, &bank_tensors());
            let err = GraftBank::from_bytes(&bytes).unwrap_err();
            assert!(matches!(err, GraftError::Conformance(_)), "{site}");
            assert!(
                err.to_string().contains("not implemented by this reader"),
                "{site}: {err}"
            );
        }
    }

    #[test]
    fn missing_hook_point_is_fatal() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.hook_point");
        let bytes = build_gguf(&meta, &bank_tensors());
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("graft.hook_point missing"));
    }

    #[test]
    fn missing_spec_version_is_fatal_and_wrong_version_rejected() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.spec_version");
        let bytes = build_gguf(&meta, &bank_tensors());
        assert!(matches!(
            GraftBank::from_bytes(&bytes),
            Err(GraftError::Conformance(_))
        ));
        let mut meta = base_meta();
        meta[1] = ("graft.spec_version", MetaValue::U32(2));
        let bytes = build_gguf(&meta, &bank_tensors());
        assert!(matches!(
            GraftBank::from_bytes(&bytes),
            Err(GraftError::Conformance(_))
        ));
    }

    #[test]
    fn missing_or_unknown_kind_is_fatal() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.kind");
        let bytes = build_gguf(&meta, &bank_tensors());
        assert!(matches!(
            GraftBank::from_bytes(&bytes),
            Err(GraftError::Conformance(_))
        ));
        let mut meta = base_meta();
        meta[3] = ("graft.kind", MetaValue::Str("magic".into()));
        let bytes = build_gguf(&meta, &bank_tensors());
        assert!(matches!(
            GraftBank::from_bytes(&bytes),
            Err(GraftError::Conformance(_))
        ));
    }

    #[test]
    fn all_three_kinds_load() {
        for (value, expected) in [
            ("prefill_kv", GraftKind::PrefillKv),
            ("softprompt_kv", GraftKind::SoftPromptKv),
            ("direct_kv", GraftKind::DirectKv),
        ] {
            let mut meta = base_meta();
            meta[3] = ("graft.kind", MetaValue::Str(value.into()));
            let bytes = build_gguf(&meta, &bank_tensors());
            let bank = GraftBank::from_bytes(&bytes).unwrap();
            assert_eq!(bank.kind, expected, "kind {value} must parse");
        }
    }

    #[test]
    fn zero_slot_canary_loads_only_without_tensors() {
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.n_slots");
        meta.push(("graft.n_slots", MetaValue::U32(0)));
        // No tensors at all: the canary artifact is a legitimately
        // zero-tensor GGUF (the generic GGUF guard allows it; the
        // graft gates above define what emptiness means).
        let bytes = build_gguf(&meta, &[]);
        let bank = GraftBank::from_bytes(&bytes).unwrap();
        assert_eq!(bank.n_slots, 0);
        assert!(bank.layers.is_empty());
        assert_eq!(bank.n_kv_heads, 0);
        assert_eq!(bank.head_dim, 0);

        // A canary with K/V tensors present is fatal.
        let bytes = build_gguf(&meta, &bank_tensors());
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("canary"));
    }

    #[test]
    fn n_slots_mismatch_with_tensor_dim_is_fatal() {
        // meta says 3 slots, tensors carry 2.
        let mut meta = base_meta();
        meta.retain(|(k, _)| *k != "graft.n_slots");
        meta.push(("graft.n_slots", MetaValue::U32(3)));
        let bytes = build_gguf(&meta, &bank_tensors());
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("disagrees with graft.n_slots"));
    }

    #[test]
    fn missing_v_for_a_k_is_fatal_and_vice_versa() {
        let dims = vec![2u64, 1, 2];
        for torn in [
            vec![
                ("graft.k.3", dims.clone(), vec![1.0, 2.0, 3.0, 4.0]),
                ("graft.v.7", dims.clone(), vec![5.0, 6.0, 7.0, 8.0]),
            ],
            vec![
                ("graft.k.3", dims.clone(), vec![1.0, 2.0, 3.0, 4.0]),
                ("graft.v.3", dims.clone(), vec![5.0, 6.0, 7.0, 8.0]),
                ("graft.k.7", dims.clone(), vec![9.0, 10.0, 11.0, 12.0]),
            ],
        ] {
            let bytes = build_gguf(&base_meta(), &torn);
            let err = GraftBank::from_bytes(&bytes).unwrap_err();
            assert!(
                err.to_string().contains("no graft.v") || err.to_string().contains("no graft.k"),
                "torn file must name the missing side: {err}"
            );
        }
    }

    #[test]
    fn no_graft_tensors_is_fatal() {
        let bytes = build_gguf(&base_meta(), &[("token_embd.weight", vec![1], vec![1.0])]);
        assert!(matches!(
            GraftBank::from_bytes(&bytes),
            Err(GraftError::Conformance(_))
        ));
    }

    #[test]
    fn wrong_dims_count_and_non_fp32_are_fatal() {
        // The fixture writer always emits ggml type 0 (F32), so the
        // dtype gate is exercised by asserting the 3-D shape gate: a 1-D
        // tensor named like a graft tensor is rejected.
        let bytes = build_gguf(
            &base_meta(),
            &[("graft.k.3", vec![4], vec![1.0, 2.0, 3.0, 4.0])],
        );
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("must be 3-D"));
    }

    #[test]
    fn head_geometry_must_be_consistent_across_tensors() {
        let mut tensors = bank_tensors();
        // graft.v.7 with 2 KV heads instead of 1.
        tensors[3] = (
            "graft.v.7",
            vec![2u64, 2, 2],
            vec![13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0],
        );
        let bytes = build_gguf(&base_meta(), &tensors);
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("n_kv_heads"));
    }

    #[test]
    fn duplicate_tensor_is_fatal() {
        let dims = vec![2u64, 1, 2];
        let bytes = build_gguf(
            &base_meta(),
            &[
                ("graft.k.3", dims.clone(), vec![1.0, 2.0, 3.0, 4.0]),
                ("graft.v.3", dims.clone(), vec![5.0, 6.0, 7.0, 8.0]),
                ("graft.k.3", dims, vec![9.0, 10.0, 11.0, 12.0]),
            ],
        );
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn missing_rope_identity_is_fatal() {
        for key in ["graft.rope_theta", "graft.rotary_dim"] {
            let mut meta = base_meta();
            meta.retain(|(k, _)| *k != key);
            let bytes = build_gguf(&meta, &bank_tensors());
            let err = GraftBank::from_bytes(&bytes).unwrap_err();
            assert!(
                err.to_string().contains(&format!("{key} missing")),
                "{key}: {err}"
            );
        }
        // Non-finite / non-positive theta is malformed.
        let mut meta = base_meta();
        meta[5] = ("graft.rope_theta", MetaValue::F32(0.0));
        let bytes = build_gguf(&meta, &bank_tensors());
        assert!(matches!(GraftBank::from_bytes(&bytes), Err(GraftError::Malformed(_))));
    }

    #[test]
    fn optional_metadata_round_trips() {
        let mut meta = base_meta();
        meta.push(("graft.position_base", MetaValue::U32(0)));
        meta.push(("graft.mrope_interleaved", MetaValue::Bool(true)));
        meta.push(("graft.quant_lane", MetaValue::Str("bf16".into())));
        let bytes = build_gguf(&meta, &bank_tensors());
        let bank = GraftBank::from_bytes(&bytes).unwrap();
        assert_eq!(bank.position_base, 0);
        assert!(bank.mrope_interleaved);
        assert_eq!(bank.quant_lane.as_deref(), Some("bf16"));
    }

    #[test]
    fn content_hash_verifies_layers_ascending_k_then_v() {
        // Independently computed SHA-256 (python3 hashlib) over the
        // fixture bank's raw LE F32 bytes in layer-ascending K,V order:
        // k.3=[1,2,3,4] v.3=[5,6,7,8] k.7=[9,10,11,12] v.7=[13,14,15,16].
        let hash = "994294717e9222764d03686b675546d724767d179b55b0a17aa99e024ac5b725";
        let mut meta = base_meta();
        meta.push(("graft.content_sha256", MetaValue::Str(hash.to_uppercase())));
        // Tensor file order must not matter: numeric layer order defines
        // the hash. Reorder the tensors relative to bank_tensors().
        let dims = vec![2u64, 1, 2];
        let reordered = vec![
            ("graft.v.7", dims.clone(), vec![13.0, 14.0, 15.0, 16.0]),
            ("graft.k.7", dims.clone(), vec![9.0, 10.0, 11.0, 12.0]),
            ("graft.v.3", dims.clone(), vec![5.0, 6.0, 7.0, 8.0]),
            ("graft.k.3", dims, vec![1.0, 2.0, 3.0, 4.0]),
        ];
        let bytes = build_gguf(&meta, &reordered);
        let bank = GraftBank::from_bytes(&bytes).unwrap();
        assert_eq!(bank.content_sha256.as_deref(), Some(hash));
    }

    #[test]
    fn content_hash_rejects_changed_tensor_bytes() {
        let hash = "994294717e9222764d03686b675546d724767d179b55b0a17aa99e024ac5b725";
        let mut meta = base_meta();
        meta.push(("graft.content_sha256", MetaValue::Str(hash.into())));
        let mut tensors = bank_tensors();
        tensors[1] = (
            "graft.v.3",
            vec![2u64, 1, 2],
            vec![5.0, 6.0, 7.0, 8.5],
        );
        let bytes = build_gguf(&meta, &tensors);
        let error = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("content_sha256 mismatch"));
    }

    #[test]
    fn malformed_content_hash_is_fatal_when_declared() {
        for value in [
            MetaValue::U32(1),
            MetaValue::Str("a".repeat(63)),
            MetaValue::Str("g".repeat(64)),
        ] {
            let mut meta = base_meta();
            meta.push(("graft.content_sha256", value));
            let bytes = build_gguf(&meta, &bank_tensors());
            assert!(GraftBank::from_bytes(&bytes)
                .unwrap_err()
                .to_string()
                .contains("content_sha256"));
        }
    }

    #[test]
    fn non_numeric_layer_suffix_is_fatal() {
        let dims = vec![2u64, 1, 2];
        let bytes = build_gguf(
            &base_meta(),
            &[
                ("graft.k.three", dims.clone(), vec![1.0, 2.0, 3.0, 4.0]),
                ("graft.v.3", dims, vec![5.0, 6.0, 7.0, 8.0]),
            ],
        );
        let err = GraftBank::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("non-numeric layer suffix"));
    }
}
