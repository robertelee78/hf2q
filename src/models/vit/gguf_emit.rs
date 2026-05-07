//! Standalone GGUF emitter for mmproj — ADR-012 Decision 18.
//!
//! Writes a valid GGUF v3 file with `general.architecture = "clip"`
//! metadata + F16 vision + projector tensors. The file is consumable
//! by llama.cpp's mtmd-cli (as a read-only reader per sovereignty)
//! and by our own `src/inference/vision/mmproj.rs` loader (Layer B's
//! round-trip gate).
//!
//! GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! Byte layout (version 3):
//!
//!   [0..4]          magic "GGUF"
//!   [4..8]          version u32
//!   [8..16]         tensor_count u64
//!   [16..24]        kv_count u64
//!   [24..]          N × metadata KV
//!   [...]           N × tensor_info (name, n_dims, shape, dtype, offset)
//!   [aligned 32]    concatenated tensor data

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use super::config::VisionConfig;
use super::convert::VitTensor;
use super::VitConvertError;

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_F32: u32 = 0;
const ALIGNMENT: u64 = 32;

/// GGUF metadata value types (subset used by mmproj writer).
///
/// Wedge-4f (iter-224 row 6): added `Bool` + `ArrayBool` so the writer
/// can emit Qwen3-VL's `clip.use_gelu` (Bool) and
/// `clip.vision.is_deepstack_layers` (Bool[]) keys at the same byte
/// shape llama.cpp's `add_vision_is_deepstack_layers` writer at
/// `/opt/llama.cpp/gguf-py/gguf/gguf_writer.py:1219-1220` produces and
/// hf2q's loader at `src/inference/vision/mmproj.rs::read_deepstack_indexes`
/// consumes.
#[derive(Debug, Clone)]
enum MetaValue {
    String(String),
    Uint32(u32),
    Float32(f32),
    Bool(bool),
    ArrayBool(Vec<bool>),
    ArrayFloat32(Vec<f32>),
}

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

/// Write a mmproj GGUF file to `output` carrying `tensors` with the
/// metadata describing `vision_config`. Tensors are written in sorted
/// name order for reproducibility.
pub fn write_mmproj_gguf(
    output: &Path,
    vision_config: &VisionConfig,
    tensors: &HashMap<String, VitTensor>,
) -> Result<(), VitConvertError> {
    let file = File::create(output)
        .map_err(|e| VitConvertError::GgufEmit(format!("create {:?}: {}", output, e)))?;
    let mut w = BufWriter::new(file);

    // Sort tensor names for deterministic output (same-input byte-identical).
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let tensor_count = names.len() as u64;

    // Metadata per clip-model.h conventions.
    let metadata = build_metadata(vision_config);
    let kv_count = metadata.len() as u64;

    // --- Header ---
    w.write_all(&GGUF_MAGIC)?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;
    w.write_all(&tensor_count.to_le_bytes())?;
    w.write_all(&kv_count.to_le_bytes())?;

    // --- Metadata KV ---
    for (key, value) in &metadata {
        write_kv(&mut w, key, value)?;
    }

    // --- Pass 1: compute tensor data offsets (writes info entries) ---
    // We need offsets relative to the start of the tensor data section.
    // Compute sizes first, then write headers + offsets, then data.
    //
    // Each tensor's F16 payload size = numel * 2.
    struct Info<'a> {
        name: &'a str,
        shape: &'a [usize],
        dtype: u32,
        size: u64,
        offset: u64,
    }

    let mut infos: Vec<Info> = Vec::with_capacity(names.len());
    let mut running_offset: u64 = 0;
    for name in &names {
        let t = &tensors[*name];
        let numel: u64 = t.shape.iter().product::<usize>() as u64;
        // ADR-021 iter-11b: honor per-tensor dtype set by the converter
        // (norms + biases are F32; weights are F16). Pre-fix this loop
        // hardcoded F16 + 2-byte size, which produced 210 wrong-dtype
        // tensors and crashed stock llama.cpp's `clip_model_loader::warmup`
        // on `GGML_ASSERT(a->type == GGML_TYPE_F32)` at ggml.c:4989.
        let (gguf_dtype, bytes_per_elem) = match t.dtype {
            crate::ir::DType::F32 => (GGML_TYPE_F32, 4u64),
            crate::ir::DType::F16 => (GGML_TYPE_F16, 2u64),
            ref other => {
                return Err(VitConvertError::GgufEmit(format!(
                    "write_mmproj_gguf: unsupported dtype {:?} for tensor {:?} \
                     — only F32 (norms + biases) and F16 (weights) are emitted",
                    other, name
                )));
            }
        };
        let size = numel * bytes_per_elem;
        infos.push(Info {
            name: name.as_str(),
            shape: &t.shape,
            dtype: gguf_dtype,
            size,
            offset: running_offset,
        });
        // Align each tensor's payload to 32 bytes.
        running_offset = align_up(running_offset + size, ALIGNMENT);
    }

    for info in &infos {
        // name (gguf-string: u64 len, utf8 bytes)
        write_gguf_string(&mut w, info.name)?;
        // n_dims u32
        w.write_all(&(info.shape.len() as u32).to_le_bytes())?;
        // shape u64 × n_dims (reverse to innermost-first per GGUF convention)
        for dim in info.shape.iter().rev() {
            w.write_all(&(*dim as u64).to_le_bytes())?;
        }
        // dtype u32
        w.write_all(&info.dtype.to_le_bytes())?;
        // offset u64 (relative to start of tensor data section)
        w.write_all(&info.offset.to_le_bytes())?;
    }

    // --- Align to 32 bytes before tensor data ---
    let header_end = current_pos(&mut w)?;
    let data_start = align_up(header_end, ALIGNMENT);
    for _ in header_end..data_start {
        w.write_all(&[0u8])?;
    }

    // --- Pass 2: write tensor data with inter-tensor alignment ---
    for (i, info) in infos.iter().enumerate() {
        let t = &tensors[info.name];
        w.write_all(&t.data)?;
        // Align to next tensor's offset boundary unless last.
        if i + 1 < infos.len() {
            let next_expected = infos[i + 1].offset;
            let written = info.offset + info.size;
            for _ in written..next_expected {
                w.write_all(&[0u8])?;
            }
        }
    }

    w.flush()
        .map_err(|e| VitConvertError::GgufEmit(format!("flush: {}", e)))?;
    Ok(())
}

fn align_up(n: u64, to: u64) -> u64 {
    let r = n % to;
    if r == 0 {
        n
    } else {
        n + (to - r)
    }
}

fn current_pos(w: &mut BufWriter<File>) -> std::io::Result<u64> {
    // BufWriter<File> implements Seek (transitive over W: Write + Seek), and
    // `stream_position()` flushes any buffered bytes before delegating to the
    // underlying file's position cursor. Without this real implementation the
    // 32-byte tensor-data alignment padding at lines 138-143 wrote ZERO bytes
    // (header_end was always 0 → align_up(0, 32) = 0 → empty 0..0 range), so
    // every emitted mmproj GGUF was 1 byte short and every tensor's data was
    // misaligned by one byte. Bug discovered 2026-05-07 via Wedge-4f live
    // smoke test on Qwen3-VL-2B-Instruct against ADR-021's worktree.
    w.stream_position()
}

/// Write a GGUF string: u64 length prefix + UTF-8 bytes.
fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

/// Write one KV pair: key-string + value-type-u32 + value bytes.
fn write_kv<W: Write>(w: &mut W, key: &str, value: &MetaValue) -> std::io::Result<()> {
    write_gguf_string(w, key)?;
    match value {
        MetaValue::String(s) => {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            write_gguf_string(w, s)
        }
        MetaValue::Uint32(v) => {
            w.write_all(&GGUF_TYPE_UINT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())
        }
        MetaValue::Float32(v) => {
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())
        }
        // Wedge-4f: Bool + ArrayBool encode as 1-byte (0|1) values per
        // GGUF spec. The mmproj loader at
        // `src/inference/vision/mmproj.rs::read_deepstack_indexes`
        // walks an `Array(Bool)` of length `block_count` and pushes
        // every `true`-flagged index into `deepstack_indexes`.
        MetaValue::Bool(v) => {
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&[*v as u8])
        }
        MetaValue::ArrayBool(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &b in arr {
                w.write_all(&[b as u8])?;
            }
            Ok(())
        }
        MetaValue::ArrayFloat32(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
    }
}

/// Build the mmproj metadata key-value list.
///
/// Keys per clip-model.h / `src/inference/vision/mmproj.rs` (load side).
///
/// Wedge-4f (iter-224 row 6) adds the Qwen3-VL-specific keys when
/// `cfg.is_qwen3vl()` returns true:
///   - `clip.use_gelu = true`               (Bool; clip-impl.h:USE_GELU)
///   - `clip.vision.spatial_merge_size`     (u32; typically 2 for Qwen3-VL)
///   - `clip.vision.is_deepstack_layers`    (Bool[block_count])
///   - `clip.vision.projection_dim`         (u32; LM hidden_size — when known)
///
/// These keys mirror the canonical writer at
/// `/opt/llama.cpp/convert_hf_to_gguf.py:4879-4896`
/// (`Qwen3VLVisionModel.set_gguf_parameters`).
fn build_metadata(cfg: &VisionConfig) -> Vec<(String, MetaValue)> {
    let mut kvs: Vec<(String, MetaValue)> = vec![
        (
            "general.architecture".into(),
            MetaValue::String("clip".into()),
        ),
        (
            "general.name".into(),
            MetaValue::String("hf2q-mmproj".into()),
        ),
        // ADR-021 iter-11b: stock llama.cpp's `clip_model_loader::get_bool`
        // (clip.cpp:2744) calls `gguf_get_val_bool` which asserts the
        // metadata type is Bool, not Uint32. Pre-fix these were emitted
        // as Uint32(0/1) and crashed `llama-mtmd-cli` on
        // `GGML_ASSERT(type_to_gguf_type<T>::value == type)`. Surfaced
        // 2026-05-07 by the live peer-reference run.
        ("clip.has_vision_encoder".into(), MetaValue::Bool(true)),
        ("clip.has_text_encoder".into(), MetaValue::Bool(false)),
        (
            "clip.projector_type".into(),
            MetaValue::String(cfg.projector_type.clone()),
        ),
        (
            "clip.vision.image_size".into(),
            MetaValue::Uint32(cfg.image_size),
        ),
        (
            "clip.vision.patch_size".into(),
            MetaValue::Uint32(cfg.patch_size),
        ),
        (
            "clip.vision.embedding_length".into(),
            MetaValue::Uint32(cfg.hidden_size),
        ),
        (
            "clip.vision.feed_forward_length".into(),
            MetaValue::Uint32(cfg.intermediate_size),
        ),
        (
            "clip.vision.attention.head_count".into(),
            MetaValue::Uint32(cfg.num_attention_heads),
        ),
        (
            "clip.vision.block_count".into(),
            MetaValue::Uint32(cfg.num_hidden_layers),
        ),
        (
            "clip.vision.attention.layer_norm_epsilon".into(),
            MetaValue::Float32(cfg.layer_norm_eps),
        ),
        (
            "clip.vision.image_mean".into(),
            MetaValue::ArrayFloat32(cfg.image_mean.to_vec()),
        ),
        (
            "clip.vision.image_std".into(),
            MetaValue::ArrayFloat32(cfg.image_std.to_vec()),
        ),
    ];

    // Optional `projection_dim`: emit only when present in
    // VisionConfig (HF source had `vision_config.projection_dim` OR
    // caller filled it from text_config.hidden_size). The mmproj
    // loader treats this key as Optional too (via
    // `gguf.metadata_u32(...)` without ok_or_else), so omitting it
    // is byte-compatible with non-Qwen3-VL profiles.
    if let Some(pd) = cfg.projection_dim {
        kvs.push((
            "clip.vision.projection_dim".into(),
            MetaValue::Uint32(pd),
        ));
    }

    // ---- Qwen3-VL extension keys -------------------------------------
    //
    // Emitted only when `cfg.is_qwen3vl()` returns true. Mirrors the
    // canonical writer's `Qwen3VLVisionModel.set_gguf_parameters` at
    // `/opt/llama.cpp/convert_hf_to_gguf.py:4879-4896`. Three keys:
    //
    //   * `clip.use_gelu = true` — `add_vision_use_gelu(True)` at
    //     line 4884; the gguf_writer helper at gguf_writer.py:1181-1182
    //     emits a top-level `clip.use_gelu` (NOT `clip.vision.use_gelu`)
    //     per `Keys.ClipVision.USE_GELU = "clip.use_gelu"` at
    //     `/opt/llama.cpp/gguf-py/gguf/constants.py:316`. Selects the
    //     ViT activation function (Qwen3-VL's MLP uses GELU; Gemma 4
    //     uses approximate-tanh GELU but emits the same Bool key).
    //
    //   * `clip.vision.spatial_merge_size` — the 2×2 patch-merger
    //     degree per `add_vision_spatial_merge_size` at line 4889.
    //     Required for Qwen3-VL because the merger fuses each 2×2
    //     patch group into a single token (4× token reduction) before
    //     the cross-modal projector. Loader reads via
    //     `MmprojConfig::from_gguf::spatial_merge_size`.
    //
    //   * `clip.vision.is_deepstack_layers` — length-block_count
    //     `Bool[]` array per `add_vision_is_deepstack_layers` at
    //     line 4895-4896. Each `true` entry flags a layer whose
    //     ViT hidden state is fed back into the LM as DeepStack
    //     augmentation. Qwen3-VL-2B uses `[5, 11, 17]` → a length-32
    //     bool array with three trues. Loader walks via
    //     `read_deepstack_indexes` at mmproj.rs:266-308 and converts
    //     to `Vec<u32>` of true-flagged indexes.
    if cfg.is_qwen3vl() {
        kvs.push((
            "clip.use_gelu".into(),
            MetaValue::Bool(true),
        ));
        if let Some(sms) = cfg.spatial_merge_size {
            kvs.push((
                "clip.vision.spatial_merge_size".into(),
                MetaValue::Uint32(sms),
            ));
        }
        if let Some(is_ds_bools) = cfg.build_is_deepstack_layers() {
            // Only emit when `deepstack_visual_indexes` was non-`None`
            // in the HF source. Empty index list → all-false array of
            // length block_count, which the loader accepts as
            // `Some(vec![])` — equivalent to "no flagged layers".
            kvs.push((
                "clip.vision.is_deepstack_layers".into(),
                MetaValue::ArrayBool(is_ds_bools),
            ));
        }
    }

    kvs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    fn tiny_config() -> VisionConfig {
        VisionConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            patch_size: 4,
            image_size: 16,
            intermediate_size: 128,
            layer_norm_eps: 1e-5,
            projector_type: "mlp".into(),
            projection_dim: None,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            // Wedge-4f: Qwen3-VL extension fields default to None for
            // non-Qwen3-VL profiles (CLIP-classic / Gemma 4).
            spatial_merge_size: None,
            deepstack_visual_indexes: None,
            temporal_patch_size: None,
        }
    }

    fn tiny_tensors() -> HashMap<String, VitTensor> {
        let mut m = HashMap::new();
        m.insert(
            "v.patch_embd.weight".into(),
            VitTensor {
                gguf_name: "v.patch_embd.weight".into(),
                shape: vec![64, 3, 4, 4],
                dtype: DType::F16,
                data: vec![0u8; 64 * 3 * 4 * 4 * 2],
            },
        );
        m.insert(
            "mm.0.weight".into(),
            VitTensor {
                gguf_name: "mm.0.weight".into(),
                shape: vec![64, 64],
                dtype: DType::F16,
                data: vec![0u8; 64 * 64 * 2],
            },
        );
        m
    }

    #[test]
    fn writes_valid_gguf_magic_and_version() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("tiny.mmproj.gguf");
        write_mmproj_gguf(&out, &tiny_config(), &tiny_tensors()).expect("write");

        let bytes = std::fs::read(&out).expect("read");
        assert!(bytes.len() > 24, "file too small");
        assert_eq!(&bytes[0..4], &GGUF_MAGIC, "bad magic");
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, GGUF_VERSION);
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tensor_count, 2);
        let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        // Use dynamic count so Wedge-4f's optional Qwen3-VL keys
        // (use_gelu / spatial_merge_size / is_deepstack_layers /
        // projection_dim) don't break this test. CLIP-classic config
        // emits 14 entries; Qwen3-VL adds 1-3 more depending on which
        // optional fields are present. The base writer-vs-build_metadata
        // self-consistency anchor is `kv_count_in_header_matches_metadata_list`
        // below — this test only pins that the header writes a sane
        // (non-zero, ≥ baseline) count.
        let expected = build_metadata(&tiny_config()).len() as u64;
        assert_eq!(kv_count, expected);
        assert!(expected >= 14, "kv_count regression below baseline");
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn write_gguf_string_format_matches_spec() {
        let mut buf = Vec::new();
        write_gguf_string(&mut buf, "hello").unwrap();
        // u64 length + utf8 bytes
        assert_eq!(&buf[0..8], &5u64.to_le_bytes());
        assert_eq!(&buf[8..13], b"hello");
    }

    #[test]
    fn build_metadata_matches_mmproj_loader_keys() {
        let cfg = tiny_config();
        let md = build_metadata(&cfg);
        let keys: Vec<&str> = md.iter().map(|(k, _)| k.as_str()).collect();

        // Required by src/inference/vision/mmproj.rs::from_gguf (the load side):
        let required = [
            "general.architecture",
            "clip.vision.image_size",
            "clip.vision.patch_size",
            "clip.vision.embedding_length",
            "clip.vision.feed_forward_length",
            "clip.vision.attention.head_count",
            "clip.vision.block_count",
        ];
        for r in &required {
            assert!(keys.contains(r), "missing required metadata key: {}", r);
        }
    }

    #[test]
    fn kv_count_in_header_matches_metadata_list() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("match-count.gguf");
        write_mmproj_gguf(&out, &tiny_config(), &tiny_tensors()).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let md = build_metadata(&tiny_config());
        assert_eq!(kv_count as usize, md.len());
    }

    #[test]
    fn empty_tensor_map_still_produces_valid_header() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("empty.gguf");
        let empty_map: HashMap<String, VitTensor> = HashMap::new();
        write_mmproj_gguf(&out, &tiny_config(), &empty_map).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[0..4], &GGUF_MAGIC);
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tensor_count, 0);
    }

    #[allow(dead_code)]
    fn _unused_f32_type() {
        let _ = GGML_TYPE_F32;
    }

    // ----- Wedge-4f (iter-224 row 6) — Qwen3-VL metadata gating -----

    fn qwen3vl_tiny_config() -> VisionConfig {
        VisionConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            patch_size: 4,
            image_size: 32,
            intermediate_size: 128,
            layer_norm_eps: 1e-5,
            projector_type: "qwen3vl_merger".into(),
            projection_dim: Some(2048),
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            spatial_merge_size: Some(2),
            deepstack_visual_indexes: Some(vec![0, 1]),
            temporal_patch_size: Some(2),
        }
    }

    /// Wedge-4f: when `cfg.is_qwen3vl()` is true, emit the three
    /// extension keys (use_gelu / spatial_merge_size /
    /// is_deepstack_layers) plus the optional projection_dim.
    #[test]
    fn wedge4f_build_metadata_emits_qwen3vl_keys_when_family_set() {
        let cfg = qwen3vl_tiny_config();
        let md = build_metadata(&cfg);
        let keys: Vec<&str> = md.iter().map(|(k, _)| k.as_str()).collect();

        for key in &[
            "clip.use_gelu",
            "clip.vision.spatial_merge_size",
            "clip.vision.is_deepstack_layers",
            "clip.vision.projection_dim",
        ] {
            assert!(
                keys.contains(key),
                "Wedge-4f: build_metadata must emit Qwen3-VL key {:?} \
                 when cfg.is_qwen3vl(); got keys: {:?}",
                key, keys
            );
        }

        // projector_type pinned to canonical Qwen3-VL string.
        let pt = md
            .iter()
            .find(|(k, _)| k == "clip.projector_type")
            .map(|(_, v)| v);
        match pt {
            Some(MetaValue::String(s)) => {
                assert_eq!(s, "qwen3vl_merger");
            }
            other => panic!("unexpected projector_type value: {:?}", other),
        }

        // is_deepstack_layers length must equal block_count = 2.
        let is_ds = md
            .iter()
            .find(|(k, _)| k == "clip.vision.is_deepstack_layers")
            .map(|(_, v)| v);
        match is_ds {
            Some(MetaValue::ArrayBool(bools)) => {
                assert_eq!(bools.len(), 2, "length must equal block_count");
                assert!(bools[0], "deepstack_visual_indexes[0]=0 → bools[0]=true");
                assert!(bools[1], "deepstack_visual_indexes[1]=1 → bools[1]=true");
            }
            other => panic!(
                "is_deepstack_layers must be ArrayBool; got {:?}",
                other
            ),
        }

        // use_gelu must be Bool(true).
        let ug = md
            .iter()
            .find(|(k, _)| k == "clip.use_gelu")
            .map(|(_, v)| v);
        match ug {
            Some(MetaValue::Bool(true)) => {}
            other => panic!("use_gelu must be Bool(true); got {:?}", other),
        }
    }

    /// Wedge-4f regression gate: a CLIP-classic config (no Qwen3-VL
    /// markers) must NOT emit any Qwen3-VL extension keys. The writer
    /// is family-gated, not always-on.
    #[test]
    fn wedge4f_build_metadata_omits_qwen3vl_keys_for_clip_classic() {
        let cfg = tiny_config(); // projector="mlp", no deepstack
        let md = build_metadata(&cfg);
        let keys: Vec<&str> = md.iter().map(|(k, _)| k.as_str()).collect();

        for key in &[
            "clip.use_gelu",
            "clip.vision.spatial_merge_size",
            "clip.vision.is_deepstack_layers",
        ] {
            assert!(
                !keys.contains(key),
                "Wedge-4f regression: CLIP-classic build_metadata must NOT \
                 emit Qwen3-VL key {:?}; got keys: {:?}",
                key, keys
            );
        }

        // CLIP-classic has no projection_dim either (since
        // tiny_config().projection_dim is None).
        assert!(
            !keys.contains(&"clip.vision.projection_dim"),
            "projection_dim is gated on Some(_) — should be absent"
        );
    }

    /// Wedge-4f writer-vs-loader round-trip: write a Qwen3-VL config
    /// to disk, read it back via `mlx_native::gguf::GgufFile`, and
    /// assert every extension key has the correct typed value the
    /// `MmprojConfig::from_gguf` loader expects.
    #[test]
    fn wedge4f_qwen3vl_metadata_round_trips_through_gguf_reader() {
        use mlx_native::gguf::{GgufFile, MetadataValue};

        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("qwen3vl-roundtrip.mmproj.gguf");
        write_mmproj_gguf(&out, &qwen3vl_tiny_config(), &tiny_tensors())
            .expect("write");

        let gguf = GgufFile::open(&out).expect("open");
        assert_eq!(
            gguf.metadata_string("clip.projector_type"),
            Some("qwen3vl_merger")
        );
        assert_eq!(gguf.metadata_u32("clip.vision.spatial_merge_size"), Some(2));
        assert_eq!(gguf.metadata_u32("clip.vision.projection_dim"), Some(2048));

        // Bool(true) round-trip on use_gelu.
        match gguf.metadata("clip.use_gelu") {
            Some(MetadataValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {:?}", other),
        }

        // Array(Bool[2]) round-trip on is_deepstack_layers.
        let raw = gguf
            .metadata("clip.vision.is_deepstack_layers")
            .expect("present");
        match raw {
            MetadataValue::Array(arr) => {
                assert_eq!(arr.len(), 2);
                let mut true_count = 0usize;
                for v in arr {
                    if matches!(v, MetadataValue::Bool(true)) {
                        true_count += 1;
                    }
                }
                assert_eq!(true_count, 2, "[0,1] indexes → both true");
            }
            other => panic!("expected Array, got {:?}", other),
        }
    }

    /// Wedge-4f: the GGUF Bool serialization is 1 byte (0|1) per
    /// upstream convention. Pin the byte-shape so a refactor doesn't
    /// silently switch to a 4-byte u32 encoding.
    #[test]
    fn wedge4f_bool_kv_writes_single_byte() {
        let mut buf = Vec::new();
        write_kv(&mut buf, "test.bool", &MetaValue::Bool(true)).unwrap();
        // Layout: u64 key_len + key bytes + u32 type + 1 byte value.
        let key_len = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        assert_eq!(key_len, 9);
        assert_eq!(&buf[8..17], b"test.bool");
        let type_tag = u32::from_le_bytes(buf[17..21].try_into().unwrap());
        assert_eq!(type_tag, GGUF_TYPE_BOOL);
        assert_eq!(buf[21], 1);
        assert_eq!(buf.len(), 22, "Bool KV must be exactly 22 bytes total");
    }

    /// Wedge-4f: ArrayBool serialization layout matches GGUF spec
    /// (type=Array + element-type=Bool + u64 length + N bytes).
    #[test]
    fn wedge4f_array_bool_kv_layout_matches_gguf_spec() {
        let mut buf = Vec::new();
        write_kv(
            &mut buf,
            "is_deepstack",
            &MetaValue::ArrayBool(vec![false, true, false, true]),
        )
        .unwrap();
        // u64 key_len + key + u32 ARRAY type + u32 BOOL element-type
        // + u64 length + 4 byte values.
        let key_len = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        assert_eq!(key_len, 12);
        let kv_off = 8 + 12;
        let array_tag = u32::from_le_bytes(buf[kv_off..kv_off + 4].try_into().unwrap());
        assert_eq!(array_tag, GGUF_TYPE_ARRAY);
        let elem_tag =
            u32::from_le_bytes(buf[kv_off + 4..kv_off + 8].try_into().unwrap());
        assert_eq!(elem_tag, GGUF_TYPE_BOOL);
        let len = u64::from_le_bytes(buf[kv_off + 8..kv_off + 16].try_into().unwrap());
        assert_eq!(len, 4);
        // 4 byte values follow.
        assert_eq!(buf[kv_off + 16], 0);
        assert_eq!(buf[kv_off + 17], 1);
        assert_eq!(buf[kv_off + 18], 0);
        assert_eq!(buf[kv_off + 19], 1);
    }
}
