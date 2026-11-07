//! ADR-063 Q4-DS4-RACE §6.1: CPU tensor-load parity of ds4's 0731 GGUF
//! (design `docs/research/adr066-iq2xxs-subset-design.md` §6.1, T1–T5).
//!
//! The test opens the file through `GgufFile` and `Deepseek4Weights` exactly
//! as serve does (mapped raw weights, F32 expansion of elementwise state) and
//! allocates no model runtime. It reads the whole 87 GB file once and about
//! 10 GB of hf2q's artifact, so it is a heavy-disk job for a reserved window.
//! It lives in the binary crate because the DeepSeek-V4 modules are
//! binary-private (`src/lib.rs` exposes only `serve::kv_persist`).
//!
//! Run (T2's independent digest comes from `scripts/four_agent/ds4_file_digest.py`):
//! ```text
//! HF2Q_DS4_FILE=<ds4 gguf> HF2Q_DS4_DIGEST_JSON=<digest json> \
//! HF2Q_DS4_HF2Q_ARTIFACT=<hf2q DeepSeek gguf> HF2Q_DS4_PARITY_RECEIPT=<out json> \
//! cargo test --release --bin hf2q ds4_file_tensor_load_parity -- --ignored --nocapture
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use mlx_native::gguf::{
    test_only_compute_byte_len, test_only_dequantize, GgufFile, MetadataValue, TensorInfo,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::quant::iq_tables::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use mlx_native::{DType, MlxBuffer, MlxDevice};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::residency::Deepseek4Weights;
use super::weights::{required_tensor_specs, TensorRole};
use super::Deepseek4Config;

/// ADR-063 Q4-DS4: 86,720,111,488 bytes, re-hashed 2026-11-02.
const DS4_FILE_SHA256: &str = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0";
const DS4_FILE_BYTES: u64 = 86_720_111_488;
/// Design §3.3, sha256 of the little-endian `iq2xxs_grid` bytes of the peer.
const IQ2XXS_GRID_SHA256: &str = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54";
const KSIGNS_IQ2XS_SHA256: &str =
    "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31";
const KMASK_IQ2XS_SHA256: &str = "5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5";
const GGUF_ALIGNMENT: u64 = 32;
const SAMPLE_LAYERS: [usize; 5] = [0, 2, 3, 21, 42];
const T5_EXPERTS: [usize; 3] = [0, 85, 170];

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Design §2.3: the storage type of every ds4 tensor class.
pub(super) fn expected_ds4_type(name: &str) -> GgmlType {
    let suffix = name.rsplit_once("blk.").map_or(name, |(_, rest)| {
        rest.split_once('.').map_or(rest, |(_, suffix)| suffix)
    });
    match suffix {
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => GgmlType::IQ2_XXS,
        "ffn_down_exps.weight" => GgmlType::Q2_K,
        "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "ffn_down_shexp.weight"
        | "attn_q_a.weight"
        | "attn_q_b.weight"
        | "attn_kv.weight"
        | "attn_output_a.weight"
        | "attn_output_b.weight"
        | "output.weight" => GgmlType::Q8_0,
        "ffn_gate_inp.weight"
        | "hc_attn_fn.weight"
        | "hc_ffn_fn.weight"
        | "output_hc_fn.weight"
        | "attn_compressor_ape.weight"
        | "attn_compressor_gate.weight"
        | "attn_compressor_kv.weight"
        | "indexer.attn_q_b.weight"
        | "indexer.proj.weight"
        | "indexer_compressor_ape.weight"
        | "indexer_compressor_gate.weight"
        | "indexer_compressor_kv.weight"
        | "token_embd.weight" => GgmlType::F16,
        "ffn_gate_tid2eid.weight" => GgmlType::I32,
        _ => GgmlType::F32,
    }
}

/// Bytes of a resident buffer in its own element type (never an untyped
/// view of a typed allocation).
fn buffer_bytes(buffer: &MlxBuffer) -> Vec<u8> {
    match buffer.dtype() {
        DType::U8 => buffer.as_slice::<u8>().unwrap().to_vec(),
        DType::F16 => bytemuck::cast_slice(buffer.as_slice::<u16>().unwrap()).to_vec(),
        DType::F32 => bytemuck::cast_slice(buffer.as_slice::<f32>().unwrap()).to_vec(),
        DType::I32 => bytemuck::cast_slice(buffer.as_slice::<i32>().unwrap()).to_vec(),
        other => panic!("unexpected resident dtype {other:?}"),
    }
}

fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32())
        .collect()
}

fn decode(bytes: &[u8], ggml_type: GgmlType) -> Vec<f32> {
    let values = bytes.len() / ggml_type.block_bytes() as usize * ggml_type.block_values() as usize;
    let mut out = vec![0.0_f32; values];
    test_only_dequantize(bytes, ggml_type, &mut out)
        .unwrap_or_else(|error| panic!("decode {ggml_type:?}: {error}"));
    out
}

/// Hand port of the peer's `dequantize_row_iq2_xxs` (design §3.2), written
/// independently of mlx-native's implementation; only the sha-pinned tables
/// are shared.
fn oracle_iq2_xxs(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 66 * 256);
    for block in bytes.chunks_exact(66) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for ib32 in 0..8 {
            let base = 2 + 8 * ib32;
            let a0 = u32::from_le_bytes(block[base..base + 4].try_into().unwrap());
            let a1 = u32::from_le_bytes(block[base + 4..base + 8].try_into().unwrap());
            let db = d * (0.5_f32 + (a1 >> 28) as f32) * 0.25_f32;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0_f32
                    } else {
                        1.0_f32
                    };
                    out.push(db * grid[j] as f32 * sign);
                }
            }
        }
    }
    out
}

/// Hand port of the peer's `dequantize_row_q2_K` loop order.
fn oracle_q2_k(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 84 * 256);
    for block in bytes.chunks_exact(84) {
        let scales = &block[..16];
        let d = half::f16::from_le_bytes([block[80], block[81]]).to_f32();
        let min = half::f16::from_le_bytes([block[82], block[83]]).to_f32();
        let mut is = 0;
        for half_index in 0..2 {
            let q = &block[16 + 32 * half_index..16 + 32 * half_index + 32];
            let mut shift = 0;
            for _ in 0..4 {
                for part in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for l in 0..16 {
                        out.push(dl * ((q[l + 16 * part] >> shift) & 3) as f32 - ml);
                    }
                }
                shift += 2;
            }
        }
    }
    out
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn relative_l2(actual: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let (mut diff, mut norm) = (0.0_f64, 0.0_f64);
    for (a, b) in actual.iter().zip(reference) {
        diff += (*a as f64 - *b as f64).powi(2);
        norm += (*b as f64).powi(2);
    }
    (diff / norm).sqrt()
}

fn pearson(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let n = left.len() as f64;
    let mean_l = left.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mean_r = right.iter().map(|v| *v as f64).sum::<f64>() / n;
    let (mut cov, mut var_l, mut var_r) = (0.0, 0.0, 0.0);
    for (l, r) in left.iter().zip(right) {
        let (dl, dr) = (*l as f64 - mean_l, *r as f64 - mean_r);
        cov += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }
    cov / (var_l.sqrt() * var_r.sqrt())
}

/// Exact byte-range reader over a GGUF tensor section.
struct TensorReader {
    file: File,
    data_offset: u64,
}

impl TensorReader {
    fn open(path: &Path, gguf: &GgufFile) -> Self {
        Self {
            file: File::open(path)
                .unwrap_or_else(|error| panic!("open {}: {error}", path.display())),
            data_offset: gguf.tensor_data_offset(),
        }
    }

    fn read(&mut self, info: &TensorInfo, start: usize, len: usize) -> Vec<u8> {
        assert!(
            start + len <= info.byte_len,
            "{} range past tensor end",
            info.name
        );
        let mut bytes = vec![0_u8; len];
        self.file
            .seek(SeekFrom::Start(
                self.data_offset + info.offset + start as u64,
            ))
            .unwrap();
        self.file.read_exact(&mut bytes).unwrap();
        bytes
    }

    fn read_all(&mut self, info: &TensorInfo) -> Vec<u8> {
        self.read(info, 0, info.byte_len)
    }
}

/// Expert matrix geometry: `[experts, rows, columns]` outer-first.
struct ExpertGeometry {
    rows: usize,
    columns: usize,
    row_bytes: usize,
}

impl ExpertGeometry {
    fn of(info: &TensorInfo) -> Self {
        let [_, rows, columns] = info.shape[..] else {
            panic!("{} is not a 3-D expert tensor: {:?}", info.name, info.shape);
        };
        let row_bytes = columns / info.ggml_type.block_values() as usize
            * info.ggml_type.block_bytes() as usize;
        Self {
            rows,
            columns,
            row_bytes,
        }
    }

    fn row_start(&self, expert: usize, row: usize) -> usize {
        (expert * self.rows + row) * self.row_bytes
    }
}

fn sample_rows(name: &str, rows: usize) -> [usize; 3] {
    if name.ends_with("ffn_down_exps.weight") {
        [0, 2047, 4095]
    } else {
        [0, 1023, 2047]
    }
    .map(|row| {
        assert!(row < rows, "{name}: sample row {row} outside {rows}");
        row
    })
}

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

/// §6.4 support: hf2q's own DeepSeek artifact reaches none of the
/// Q4-DS4-RACE routes (header only). Its file type stays Q2_K (L1), its
/// pre-tokenizer `deepseek-v3` (L2), both defaulted keys are present with the
/// default values (L3), `compress_ratios` has exactly `block_count` entries
/// (L4), `token_embd` is block storage (L5), `output_hc_fn` is not float (L6),
/// no routed expert is IQ2_XXS (L7) and no raw matrix is F16 (L9).
#[test]
#[ignore = "reads hf2q's DeepSeek-V4 artifact header"]
fn hf2q_artifact_takes_none_of_the_ds4_file_routes() {
    let path = env_path("HF2Q_DEEPSEEK4_GGUF");
    let gguf = GgufFile::open(&path).expect("open hf2q artifact");
    assert_eq!(
        crate::serve::quant_select::quant_type_from_gguf_path(&path).unwrap(),
        crate::serve::quant_select::QuantType::Q2_K
    );
    assert_eq!(
        gguf.metadata_string("tokenizer.ggml.pre"),
        Some("deepseek-v3")
    );
    let cfg = Deepseek4Config::from_gguf(&gguf).unwrap();
    assert_eq!(
        gguf.metadata_u32("deepseek4.embedding_length_out"),
        Some(cfg.hidden_size_out)
    );
    assert_eq!(
        cfg.hidden_size_out,
        cfg.hidden_size * cfg.hyper_connection_count
    );
    assert!(gguf.metadata("deepseek4.swiglu_clamp_shexp").is_some());
    let Some(MetadataValue::Array(ratios)) = gguf.metadata("deepseek4.attention.compress_ratios")
    else {
        panic!("compress_ratios array");
    };
    assert_eq!(ratios.len(), cfg.num_hidden_layers as usize);
    let mut raw_types = BTreeMap::<String, usize>::new();
    for spec in required_tensor_specs(&cfg) {
        let info = gguf.tensor_info(&spec.name).unwrap();
        assert_ne!(info.ggml_type, GgmlType::IQ2_XXS, "{}", spec.name);
        if spec.role == TensorRole::RawMatrix {
            assert!(
                !matches!(info.ggml_type, GgmlType::F16 | GgmlType::BF16),
                "{} is {:?}",
                spec.name,
                info.ggml_type
            );
            *raw_types
                .entry(format!("{:?}", info.ggml_type))
                .or_default() += 1;
        }
    }
    for name in ["token_embd.weight", "output_hc_fn.weight"] {
        assert_eq!(
            gguf.tensor_info(name).unwrap().ggml_type,
            GgmlType::Q8_0,
            "{name}"
        );
    }
    eprintln!("hf2q artifact raw matrix types: {raw_types:?}");
}

#[test]
fn iq2_xxs_oracle_tables_match_the_design_pins() {
    let grid: Vec<u8> = IQ2XXS_GRID
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(sha256_hex(&grid), IQ2XXS_GRID_SHA256);
    assert_eq!(sha256_hex(&KSIGNS_IQ2XS), KSIGNS_IQ2XS_SHA256);
    assert_eq!(sha256_hex(&KMASK_IQ2XS), KMASK_IQ2XS_SHA256);
}

#[test]
fn hand_ported_oracles_match_mlx_native_on_synthetic_blocks() {
    // Deterministic blocks with positive normal scales; the real-block check
    // is T3 below.
    let mut iq2 = vec![0_u8; 66 * 16];
    let mut q2k = vec![0_u8; 84 * 16];
    for (index, byte) in iq2.iter_mut().enumerate() {
        *byte = (index * 131 + 7) as u8;
    }
    for (index, byte) in q2k.iter_mut().enumerate() {
        *byte = (index * 97 + 3) as u8;
    }
    for block in 0..16 {
        iq2[block * 66..block * 66 + 2]
            .copy_from_slice(&half::f16::from_f32(0.01 + block as f32 * 1e-3).to_le_bytes());
        q2k[block * 84 + 80..block * 84 + 82]
            .copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
        q2k[block * 84 + 82..block * 84 + 84]
            .copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
    }
    assert_eq!(
        bits(&decode(&iq2, GgmlType::IQ2_XXS)),
        bits(&oracle_iq2_xxs(&iq2))
    );
    assert_eq!(
        bits(&decode(&q2k, GgmlType::Q2_K)),
        bits(&oracle_q2_k(&q2k))
    );
}

#[test]
#[ignore = "reads all of ds4's 87 GB DeepSeek-V4 GGUF and parts of hf2q's artifact (reserved window)"]
fn ds4_file_tensor_load_parity() {
    let ds4_path = env_path("HF2Q_DS4_FILE");
    let digest_path = env_path("HF2Q_DS4_DIGEST_JSON");
    let hf2q_path = env_path("HF2Q_DS4_HF2Q_ARTIFACT");
    let receipt_path = env_path("HF2Q_DS4_PARITY_RECEIPT");
    let gguf = GgufFile::open(&ds4_path).expect("open ds4 file");
    let mut receipt = BTreeMap::<&str, Value>::new();
    receipt.insert("ds4_file", json!(ds4_path.display().to_string()));
    receipt.insert("hf2q_artifact", json!(hf2q_path.display().to_string()));
    receipt.insert("digest_json", json!(digest_path.display().to_string()));

    // ── T1: header, catalog, byte extents, loader gates L1–L4 ──────────
    assert_eq!(std::fs::metadata(&ds4_path).unwrap().len(), DS4_FILE_BYTES);
    let quant = crate::serve::quant_select::quant_type_from_gguf_path(&ds4_path).expect("L1");
    assert_eq!(quant, crate::serve::quant_select::QuantType::IQ2_XXS);
    super::tokenizer::build_tokenizer_from_gguf(&gguf).expect("L2 tokenizer");
    let cfg = Deepseek4Config::from_gguf(&gguf).expect("L3/L4 config");
    assert_eq!(cfg.compress_ratios.len(), 43);
    assert_eq!(cfg.hidden_size_out, 16_384);
    let specs = required_tensor_specs(&cfg);
    assert_eq!(specs.len(), 1_328);
    assert_eq!(gguf.tensor_names().len(), 1_328);
    let mut type_counts = BTreeMap::<String, usize>::new();
    let mut by_offset: Vec<&TensorInfo> = Vec::new();
    for spec in &specs {
        let info = gguf.tensor_info(&spec.name).expect("catalog tensor");
        assert_eq!(
            info.ggml_type,
            expected_ds4_type(&spec.name),
            "{} type",
            spec.name
        );
        assert_eq!(
            info.byte_len,
            test_only_compute_byte_len(&info.shape, info.ggml_type).unwrap(),
            "{} byte length",
            spec.name
        );
        *type_counts
            .entry(format!("{:?}", info.ggml_type))
            .or_default() += 1;
        by_offset.push(info);
    }
    by_offset.sort_by_key(|info| info.offset);
    let section_end = DS4_FILE_BYTES - gguf.tensor_data_offset();
    for (index, info) in by_offset.iter().enumerate() {
        let next = by_offset
            .get(index + 1)
            .map_or(section_end, |next| next.offset);
        let extent = next - info.offset;
        assert!(
            extent >= info.byte_len as u64 && extent - (info.byte_len as u64) < GGUF_ALIGNMENT,
            "{} extent {extent} vs geometry {}",
            info.name,
            info.byte_len
        );
    }
    let expected_counts: BTreeMap<String, usize> = [
        ("F32", 492),
        ("F16", 359),
        ("Q8_0", 345),
        ("IQ2_XXS", 86),
        ("Q2_K", 43),
        ("I32", 3),
    ]
    .into_iter()
    .map(|(name, count)| (name.to_string(), count))
    .collect();
    assert_eq!(type_counts, expected_counts);
    receipt.insert(
        "t1",
        json!({"pass": true, "tensors": 1328, "type_counts": type_counts}),
    );
    eprintln!("T1 pass");

    // ── T2: bytes as loaded equal the independent reader; file identity ─
    let digest: Value = serde_json::from_slice(&std::fs::read(&digest_path).unwrap()).unwrap();
    assert_eq!(
        digest["file_sha256"].as_str(),
        Some(DS4_FILE_SHA256),
        "full-file digest"
    );
    let independent: HashMap<String, (u64, u64, String)> = digest["tensors"]
        .as_array()
        .expect("digest tensors")
        .iter()
        .map(|entry| {
            (
                entry["name"].as_str().unwrap().to_string(),
                (
                    entry["offset"].as_u64().unwrap(),
                    entry["nbytes"].as_u64().unwrap(),
                    entry["sha256"].as_str().unwrap().to_string(),
                ),
            )
        })
        .collect();
    assert_eq!(independent.len(), 1_328);
    let device = MlxDevice::new().expect("Metal device for residency");
    let weights = Deepseek4Weights::load_from_gguf(&gguf, &cfg, device).expect("residency");
    let mut ds4_sha = HashMap::<String, String>::new();
    let mut expanded = Vec::new();
    let mut reader = TensorReader::open(&ds4_path, &gguf);
    for spec in &specs {
        let info = gguf.tensor_info(&spec.name).unwrap();
        let (offset, nbytes, want) = independent
            .get(&spec.name)
            .unwrap_or_else(|| panic!("{} missing from digest", spec.name));
        // The digest reports absolute file offsets and block-geometry sizes.
        assert_eq!(
            (*offset, *nbytes as usize),
            (gguf.tensor_data_offset() + info.offset, info.byte_len),
            "{}",
            spec.name
        );
        let resident = weights.tensor(&spec.name, spec.role).unwrap();
        let file_sha = if spec.role == TensorRole::ElementwiseF32 && info.ggml_type != GgmlType::F32
        {
            // Expanded at load: T4 checks the expansion against these bytes.
            expanded.push(spec.name.clone());
            sha256_hex(&reader.read_all(info))
        } else {
            sha256_hex(&buffer_bytes(resident))
        };
        assert_eq!(&file_sha, want, "{} bytes as loaded", spec.name);
        ds4_sha.insert(spec.name.clone(), file_sha);
    }
    receipt.insert(
        "t2",
        json!({"pass": true, "file_sha256": DS4_FILE_SHA256, "tensors": ds4_sha.len(),
               "expanded_checked_from_file_bytes": expanded}),
    );
    eprintln!("T2 pass");

    // ── T3: CPU dequant of real blocks equals the bit-exact oracle ─────
    let mut t3_rows = 0_usize;
    for layer in SAMPLE_LAYERS {
        for kind in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let name = format!("blk.{layer}.{kind}.weight");
            let info = gguf.tensor_info(&name).unwrap();
            let bytes = buffer_bytes(weights.raw_matrix(&name).unwrap());
            let geometry = ExpertGeometry::of(info);
            for expert in 0..info.shape[0] {
                for row in sample_rows(&name, geometry.rows) {
                    let start = geometry.row_start(expert, row);
                    let block = &bytes[start..start + geometry.row_bytes];
                    let oracle = match info.ggml_type {
                        GgmlType::IQ2_XXS => oracle_iq2_xxs(block),
                        GgmlType::Q2_K => oracle_q2_k(block),
                        other => panic!("{name}: unexpected expert type {other:?}"),
                    };
                    assert_eq!(
                        bits(&decode(block, info.ggml_type)),
                        bits(&oracle),
                        "{name} expert {expert} row {row}"
                    );
                    t3_rows += 1;
                }
            }
        }
    }
    receipt.insert(
        "t3",
        json!({"pass": true, "rows": t3_rows, "layers": SAMPLE_LAYERS}),
    );
    eprintln!("T3 pass ({t3_rows} rows)");

    // ── T4: F16 classes decode finite; F16 *_ape expansion is exact ────
    let mut t4_f16 = 0_usize;
    for spec in &specs {
        let info = gguf.tensor_info(&spec.name).unwrap();
        if info.ggml_type != GgmlType::F16 {
            continue;
        }
        let raw = if spec.role == TensorRole::RawMatrix {
            buffer_bytes(weights.raw_matrix(&spec.name).unwrap())
        } else {
            reader.read_all(info)
        };
        let values = f16_to_f32(&raw);
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{} non-finite",
            spec.name
        );
        if spec.role == TensorRole::ElementwiseF32 {
            let loaded = weights
                .f32_state(&spec.name)
                .unwrap()
                .as_slice::<f32>()
                .unwrap();
            assert_eq!(bits(loaded), bits(&values), "{} F32 expansion", spec.name);
        }
        t4_f16 += 1;
    }
    assert_eq!(t4_f16, 359);
    receipt.insert(
        "t4",
        json!({"pass": true, "f16_tensors": t4_f16, "expanded": expanded.len()}),
    );
    eprintln!("T4 pass");
    drop(weights);

    // ── T5: same-source cross-check against hf2q's artifact ────────────
    let hf2q = GgufFile::open(&hf2q_path).expect("open hf2q artifact");
    let mut hf2q_reader = TensorReader::open(&hf2q_path, &hf2q);
    let mut identical = 0_usize;
    let mut f16_vs_q8 = Vec::new();
    let mut router = Vec::new();
    let mut experts = Vec::new();
    let mut shared = Vec::new();
    let mut failures = Vec::new();
    for spec in &specs {
        let ds4 = gguf.tensor_info(&spec.name).unwrap();
        let other = hf2q
            .tensor_info(&spec.name)
            .unwrap_or_else(|| panic!("{} missing from hf2q artifact", spec.name));
        assert_eq!(ds4.shape, other.shape, "{} shape", spec.name);
        match (ds4.ggml_type, other.ggml_type) {
            (a, b) if a == b && matches!(a, GgmlType::F32 | GgmlType::Q8_0 | GgmlType::I32) => {
                let other_sha = sha256_hex(&hf2q_reader.read_all(other));
                if ds4_sha[&spec.name] == other_sha {
                    identical += 1;
                } else {
                    failures.push(format!("{} not byte-identical", spec.name));
                }
            }
            (GgmlType::F16, GgmlType::Q8_0) | (GgmlType::F16, GgmlType::F32) => {
                let mine = f16_to_f32(&reader.read_all(ds4));
                let theirs = decode(&hf2q_reader.read_all(other), other.ggml_type);
                let error = relative_l2(&mine, &theirs);
                let (bound, list) = if other.ggml_type == GgmlType::F32 {
                    (1e-6, &mut router)
                } else {
                    (1e-2, &mut f16_vs_q8)
                };
                if error > bound {
                    failures.push(format!("{}: relative L2 {error} > {bound}", spec.name));
                }
                list.push(json!({"name": spec.name, "relative_l2": error}));
            }
            (GgmlType::IQ2_XXS | GgmlType::Q2_K, GgmlType::Q2_K | GgmlType::Q3_K)
                if spec.name.ends_with("_exps.weight") =>
            {
                let record = expert_layout_check(
                    &spec.name,
                    ds4,
                    &mut reader,
                    &hf2q,
                    &mut hf2q_reader,
                    cfg.num_hidden_layers as usize,
                );
                if !record["pass"].as_bool().unwrap() {
                    failures.push(format!("{} expert layout rule failed", spec.name));
                }
                experts.push(record);
            }
            // Shared experts (Q8_0 against hf2q's Q2_K/Q3_K) have no §6.1
            // rule; their error is recorded descriptively.
            (GgmlType::Q8_0, GgmlType::Q2_K | GgmlType::Q3_K)
                if spec.name.ends_with("_shexp.weight") =>
            {
                let mine = decode(&reader.read_all(ds4), ds4.ggml_type);
                let theirs = decode(&hf2q_reader.read_all(other), other.ggml_type);
                shared.push(json!({"name": spec.name, "relative_l2": relative_l2(&mine, &theirs)}));
            }
            (a, b) => failures.push(format!("{}: unexpected type pair {a:?}/{b:?}", spec.name)),
        }
    }
    if experts.len() != 129 {
        failures.push(format!(
            "{} routed-expert tensors checked, expected 129",
            experts.len()
        ));
    }
    receipt.insert(
        "t5",
        json!({"pass": failures.is_empty(), "byte_identical": identical,
               "f16_vs_q8_0": f16_vs_q8, "f16_router_vs_f32": router,
               "experts": experts, "shared_experts_descriptive": shared,
               "failures": failures}),
    );
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap())
        .unwrap_or_else(|error| panic!("write receipt {}: {error}", receipt_path.display()));
    eprintln!("receipt: {}", receipt_path.display());
    assert!(failures.is_empty(), "T5 failures: {failures:#?}");
}

/// Design §6.1 T5 expert rule: the correct hf2q row correlates best with
/// ds4's decoded row and beats the best of four layout controls by ≥ 0.3.
fn expert_layout_check(
    name: &str,
    ds4: &TensorInfo,
    ds4_reader: &mut TensorReader,
    hf2q: &GgufFile,
    hf2q_reader: &mut TensorReader,
    layers: usize,
) -> Value {
    let (layer, kind) = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .map(|(layer, kind)| (layer.parse::<usize>().unwrap(), kind))
        .unwrap();
    let other = hf2q.tensor_info(name).unwrap();
    let next_layer = hf2q
        .tensor_info(&format!("blk.{}.{kind}", (layer + 1) % layers))
        .unwrap();
    let ds4_geometry = ExpertGeometry::of(ds4);
    let geometry = ExpertGeometry::of(other);
    let experts = other.shape[0];
    let expert_values = geometry.rows * geometry.columns;
    let mut candidates: [Vec<f32>; 6] = Default::default();
    for expert in T5_EXPERTS {
        let expert_bytes = geometry.rows * geometry.row_bytes;
        let matrix = decode(
            &hf2q_reader.read(other, geometry.row_start(expert, 0), expert_bytes),
            other.ggml_type,
        );
        let next_expert = decode(
            &hf2q_reader.read(
                other,
                geometry.row_start((expert + 1) % experts, 0),
                expert_bytes,
            ),
            other.ggml_type,
        );
        // The matrix transposed (`columns × rows`), flat, read at the row width.
        let mut transposed = vec![0.0_f32; expert_values];
        for row in 0..geometry.rows {
            for column in 0..geometry.columns {
                transposed[column * geometry.rows + row] = matrix[row * geometry.columns + column];
            }
        }
        for row in sample_rows(name, geometry.rows) {
            let start = ds4_geometry.row_start(expert, row);
            candidates[0].extend(decode(
                &ds4_reader.read(ds4, start, ds4_geometry.row_bytes),
                ds4.ggml_type,
            ));
            let span = row * geometry.columns..(row + 1) * geometry.columns;
            candidates[1].extend_from_slice(&matrix[span.clone()]);
            candidates[2].extend_from_slice(&next_expert[span.clone()]);
            candidates[3].extend(decode(
                &hf2q_reader.read(
                    next_layer,
                    geometry.row_start(expert, row),
                    geometry.row_bytes,
                ),
                next_layer.ggml_type,
            ));
            candidates[4].extend_from_slice(&transposed[span]);
            candidates[5].extend(
                (0..geometry.columns)
                    .map(|index| matrix[(row * geometry.columns + 256 + index) % expert_values]),
            );
        }
    }
    let [ds4_rows, correct, wrong_expert, wrong_layer, transposed, shifted] = &candidates;
    let correlation = |values: &Vec<f32>| pearson(ds4_rows, values);
    let correct = correlation(correct);
    let controls = [
        ("wrong_expert", correlation(wrong_expert)),
        ("wrong_layer", correlation(wrong_layer)),
        ("transposed", correlation(transposed)),
        ("shifted", correlation(shifted)),
    ];
    let best_control = controls
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::NEG_INFINITY, f64::max);
    let pass = correct.is_finite() && correct - best_control >= 0.3;
    json!({
        "name": name,
        "ds4_type": format!("{:?}", ds4.ggml_type),
        "hf2q_type": format!("{:?}", other.ggml_type),
        "correct": correct,
        "controls": controls.iter().map(|(name, value)| (name.to_string(), json!(value))).collect::<serde_json::Map<_, _>>(),
        "margin": correct - best_control,
        "pass": pass,
    })
}
