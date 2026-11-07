//! #726: model-free chunk-split invariance of DeepSeek-V4 matrix prefill.
//!
//! Synthetic weights at the ds4 file's storage types (IQ2_XXS gate/up and
//! Q2_K down experts, Q8_0 dense, F16 router/HC/compressor/indexer and
//! embeddings) and real layer shapes (hidden 4096, 64 heads x 512, Q/O
//! ranks 1024, 256 experts x 2048, top-6, indexer 64 x 128, top-512, full
//! vocabulary) for the file's first five layers: two window layers, CSA
//! (ratio 4, with indexer), HCA (ratio 128), CSA. Layers 0-2 are
//! hash-routed and 3-4 score-routed, as in the file.
//!
//! The same token rows are ingested through the production matrix prefill
//! entry (`forward_verifier_prefill`, one call per chunk, every chunk at
//! least `MIN_MATRIX_APPEND_TOKENS` rows) under different splits. Every
//! final-state row, every physical byte of every cache buffer (window ring,
//! compressed rows, indexer rows, the four compressor states) and the
//! last-row logits must equal the reference split bitwise:
//! - group A, 2048 rows from position 0: one 2048-row chunk;
//! - group B, 4609 rows: a reference split of 2048 + 2049 (one oversized
//!   chunk; the serving planner balance-splits a cold prompt instead), then
//!   512 rows from the unaligned start 4097.
//!
//! Cooperative cohort arms (widths 2/3/4 at the engine's 1024/640/512-row
//! sizing, companion lanes ingesting other tokens) must match too.
//!
//! The reference arm runs first and again last (run-to-run control, so a
//! split difference cannot be pool or ordering noise). A token-wise final
//! row (the engine's tail rule) must differ: proof the comparison can see a
//! numerics change. Chunks below the matrix minimum (2, 8, 32 rows) are
//! recorded, not asserted.

use std::fs::File;
use std::io::BufWriter;

use mlx_native::gguf::GgufFile;
use mlx_native::MlxBuffer;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;

use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType as WriterType;

use super::cache::Deepseek4Cache;
use super::ds4_file_parity_tests::expected_ds4_type;
use super::residency_tests::metadata;
use super::verifier_forward::MIN_MATRIX_APPEND_TOKENS;
use super::weights::{required_tensor_specs, TensorRole, TensorSpec};
use super::{Deepseek4Config, Deepseek4Model};

/// Cache capacity shared by every arm, so physical buffers compare 1:1.
const CONTEXT: usize = 8192;
const GROUP_A_ROWS: usize = 2048;
const GROUP_B_ROWS: usize = 4609;

/// The ds4 file's first five layers at its real geometry.
fn config() -> Deepseek4Config {
    Deepseek4Config {
        hidden_size: 4096,
        hidden_size_out: 16384,
        num_hidden_layers: 5,
        max_position_embeddings: 1_048_576,
        vocab_size: 129_280,
        num_attention_heads: 64,
        num_key_value_heads: 1,
        head_dim: 512,
        rope_head_dim: 64,
        rope_theta: 10000.0,
        rope_factor: 16.0,
        original_context_length: 65536,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        q_lora_rank: 1024,
        o_lora_rank: 1024,
        output_groups: 8,
        sliding_window: 128,
        compress_ratios: vec![0, 0, 4, 128, 4],
        compress_rope_theta: 160000.0,
        index_num_heads: 64,
        index_head_dim: 128,
        index_top_k: 512,
        rms_norm_eps: 1e-6,
        num_experts: 256,
        num_experts_per_tok: 6,
        num_shared_experts: 1,
        expert_intermediate_size: 2048,
        route_scale: 1.5,
        normalize_topk: true,
        swiglu_clamp_experts: vec![10.0; 5],
        swiglu_clamp_shared: vec![10.0; 5],
        hyper_connection_count: 4,
        hyper_connection_sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: 3,
    }
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn fill(&mut self, out: &mut Vec<u8>, bytes: usize) {
        let mut left = bytes;
        while left > 0 {
            let word = self.next().to_le_bytes();
            let take = left.min(8);
            out.extend_from_slice(&word[..take]);
            left -= take;
        }
    }
}

fn f16_bytes(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_le_bytes()
}

fn writer_type(ggml_type: GgmlType) -> WriterType {
    match ggml_type {
        GgmlType::F32 => WriterType::F32,
        GgmlType::F16 => WriterType::F16,
        GgmlType::Q8_0 => WriterType::Q8_0,
        GgmlType::Q2_K => WriterType::Q2_K,
        GgmlType::IQ2_XXS => WriterType::IQ2_XXS,
        GgmlType::I32 => WriterType::I32,
        other => panic!("no synthetic payload for {other:?}"),
    }
}

/// Element value of an elementwise (F32-resident) tensor: RMS-norm weights
/// near 1, HC scales near 1, a small router selection bias, and moderate
/// HC bases, attention sinks and compressor position embeddings.
fn elementwise_value(name: &str, rng: &mut SplitMix) -> f32 {
    let u = rng.signed();
    if name.contains("norm") {
        1.0 + 0.1 * u
    } else if name.ends_with("_scale.weight") {
        1.0 + 0.5 * u
    } else if name.contains("exp_probs_b") {
        0.05 * u
    } else {
        0.5 * u
    }
}

/// Append `rows` rows of `cols` elements in `ggml_type`. Raw matrices have
/// unit-variance-preserving magnitude (`rms ~ 1/sqrt(cols)`); embeddings are
/// uniform in [-1, 1).
fn payload_rows(
    spec: &TensorSpec,
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
    rng: &mut SplitMix,
    out: &mut Vec<u8>,
) {
    let rms = 1.0 / (cols as f32).sqrt();
    let uniform = rms * 3.0_f32.sqrt();
    match ggml_type {
        GgmlType::F32 => {
            for _ in 0..rows * cols {
                out.extend_from_slice(&elementwise_value(&spec.name, rng).to_le_bytes());
            }
        }
        GgmlType::F16 => {
            for _ in 0..rows * cols {
                let value = match spec.role {
                    TensorRole::ElementwiseF32 => elementwise_value(&spec.name, rng),
                    _ if spec.name == "token_embd.weight" => rng.signed(),
                    _ => uniform * rng.signed(),
                };
                out.extend_from_slice(&f16_bytes(value));
            }
        }
        GgmlType::Q8_0 => {
            for _ in 0..rows * cols / 32 {
                out.extend_from_slice(&f16_bytes(uniform / 127.0));
                for _ in 0..4 {
                    for byte in rng.next().to_le_bytes() {
                        out.push(((byte % 255) as i16 - 127) as i8 as u8);
                    }
                }
            }
        }
        GgmlType::Q2_K => {
            // scale nibble 2, min nibble 3, d = dmin: values {-3,-1,1,3} * d.
            let d = rms / 5.0_f32.sqrt();
            for _ in 0..rows * cols / 256 {
                out.extend_from_slice(&[0x32; 16]);
                rng.fill(out, 64);
                out.extend_from_slice(&f16_bytes(d));
                out.extend_from_slice(&f16_bytes(d));
            }
        }
        GgmlType::IQ2_XXS => {
            // Grid magnitudes 8/25/43 times (0.5 + 4-bit scale) / 4: rms ~ 58 d.
            let d = rms / 58.0;
            for _ in 0..rows * cols / 256 {
                out.extend_from_slice(&f16_bytes(d));
                rng.fill(out, 64);
            }
        }
        GgmlType::I32 => {
            // Hash routing: `cols` distinct experts per token row.
            for _ in 0..rows {
                let mut chosen: Vec<i32> = Vec::with_capacity(cols);
                while chosen.len() < cols {
                    let expert = (rng.next() % 256) as i32;
                    if !chosen.contains(&expert) {
                        chosen.push(expert);
                    }
                }
                for expert in chosen {
                    out.extend_from_slice(&expert.to_le_bytes());
                }
            }
        }
        other => panic!("no synthetic payload for {other:?}"),
    }
}

/// Stream the synthetic verifier GGUF tensor by tensor (~12 GB on disk).
fn write_fixture(cfg: &Deepseek4Config) -> (tempfile::TempDir, GgufFile) {
    let directory = tempfile::Builder::new()
        .prefix("hf2q-ds4-chunk-split")
        .tempdir()
        .unwrap();
    let path = directory.path().join("weights.gguf");
    let specs = required_tensor_specs(cfg);
    let metadata = metadata(cfg);
    let mut writer = GgufWriter::new(BufWriter::with_capacity(
        8 << 20,
        File::create(&path).unwrap(),
    ));
    writer
        .write_header(specs.len() as u64, metadata.len() as u64)
        .unwrap();
    for (key, value) in &metadata {
        writer.write_metadata_kv(key, value).unwrap();
    }
    let types: Vec<GgmlType> = specs
        .iter()
        .map(|spec| expected_ds4_type(&spec.name))
        .collect();
    for (spec, &ggml_type) in specs.iter().zip(&types) {
        let dims: Vec<u64> = spec.shape.iter().rev().map(|&dim| dim as u64).collect();
        writer
            .reserve_tensor_info(&spec.name, &dims, writer_type(ggml_type))
            .unwrap();
    }
    writer.pad_to_alignment().unwrap();
    let mut buffer = Vec::with_capacity(64 << 20);
    for (index, (spec, &ggml_type)) in specs.iter().zip(&types).enumerate() {
        let cols = *spec.shape.last().unwrap();
        let rows: usize = spec.shape[..spec.shape.len() - 1].iter().product();
        let row_bytes = writer_type(ggml_type).row_size(cols);
        let batch = ((32 << 20) / row_bytes).max(1);
        let mut rng = SplitMix(
            spec.name
                .bytes()
                .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
                }),
        );
        writer.begin_tensor_payload(index).unwrap();
        let mut done = 0;
        while done < rows {
            let take = batch.min(rows - done);
            buffer.clear();
            payload_rows(spec, ggml_type, take, cols, &mut rng, &mut buffer);
            assert_eq!(buffer.len(), take * row_bytes, "{}", spec.name);
            writer.write_tensor_payload_chunk(index, &buffer).unwrap();
            done += take;
        }
        writer.finish_tensor_payload(index).unwrap();
    }
    writer.finalize().unwrap();
    drop(writer);
    let gguf = GgufFile::open(&path).unwrap();
    (directory, gguf)
}

/// Every physical byte of one cache buffer, with its row width.
pub(super) struct CacheBytes {
    pub(super) label: String,
    pub(super) row_bytes: usize,
    pub(super) bytes: Vec<u8>,
}

pub(super) fn cache_bytes(cache: &Deepseek4Cache) -> Vec<CacheBytes> {
    let mut out = Vec::new();
    for (layer, buffers) in cache.layers().iter().enumerate() {
        for (name, buffer) in [
            ("attention_kv", Some(&buffers.attention_kv)),
            ("indexer_kv", buffers.indexer_kv.as_ref()),
            ("main_kv_state", buffers.main_kv_state.as_ref()),
            ("main_score_state", buffers.main_score_state.as_ref()),
            ("indexer_kv_state", buffers.indexer_kv_state.as_ref()),
            ("indexer_score_state", buffers.indexer_score_state.as_ref()),
        ] {
            let Some(buffer) = buffer else { continue };
            let row_bytes =
                buffer.shape()[1..].iter().product::<usize>() * buffer.dtype().size_of();
            out.push(CacheBytes {
                label: format!("layer {layer} {name}"),
                row_bytes,
                bytes: buffer.as_slice::<u8>().unwrap().to_vec(),
            });
        }
    }
    out
}

/// One split's observable result.
struct Capture {
    /// Final-layer state of every ingested row, `[rows, hc, hidden]` F32.
    rows: Vec<u8>,
    cache: Vec<CacheBytes>,
    logits: Vec<u8>,
}

/// Ingest `tokens` chunk by chunk. A 1-row chunk takes the token-wise
/// verifier (`forward_verifier_one`, what the engine's tail rule uses);
/// every other chunk is one matrix prefill transaction. With `width > 1`
/// every chunk instead runs as a cooperative cohort with `width - 1`
/// companion lanes ingesting different tokens at the same lengths (the
/// engine's cohort route; its FFN packs all lanes' rows into one `mm_id`).
fn run_split(
    model: &mut Deepseek4Model,
    tokens: &[u32],
    split: &[usize],
    width: usize,
) -> Capture {
    assert_eq!(split.iter().sum::<usize>(), tokens.len());
    let mut cache = model.allocate_cache(CONTEXT).unwrap();
    let mut companions: Vec<Deepseek4Cache> = (1..width)
        .map(|_| model.allocate_cache(CONTEXT).unwrap())
        .collect();
    let vocab = model.cfg.vocab_size;
    let companion_tokens: Vec<Vec<u32>> = (1..width as u32)
        .map(|lane| {
            tokens
                .iter()
                .map(|&token| (token + lane * 7919) % vocab)
                .collect()
        })
        .collect();
    let row_bytes = (model.cfg.hyper_connection_count * model.cfg.hidden_size) as usize * 4;
    let mut rows = Vec::with_capacity(tokens.len() * row_bytes);
    let mut offset = 0;
    let mut last = None;
    for &len in split {
        let state = if len == 1 {
            assert_eq!(width, 1, "token-wise rows are solo");
            model.forward_verifier_one(tokens[offset], &mut cache)
        } else if width == 1 {
            model.forward_verifier_prefill(&tokens[offset..offset + len], &mut cache)
        } else {
            let batches: Vec<&[u32]> = std::iter::once(&tokens[offset..offset + len])
                .chain(
                    companion_tokens
                        .iter()
                        .map(|lane| &lane[offset..offset + len]),
                )
                .collect();
            let mut caches: Vec<&mut Deepseek4Cache> = std::iter::once(&mut cache)
                .chain(companions.iter_mut())
                .collect();
            model
                .forward_verifier_prefill_cohort(&batches, &mut caches)
                .map(|mut states| states.swap_remove(0))
        }
        .unwrap_or_else(|error| panic!("{split:?} width {width} chunk at {offset}: {error:#}"));
        assert_eq!(state.shape()[0], len);
        rows.extend_from_slice(bytemuck::cast_slice(
            state.as_logical_slice::<f32>().unwrap(),
        ));
        offset += len;
        last = Some(state);
    }
    assert_eq!(cache.position(), tokens.len());
    let last = model.last_token_state(&last.unwrap()).unwrap();
    let logits = model.forward_logits(&last).unwrap();
    Capture {
        rows,
        cache: cache_bytes(&cache),
        logits: bytemuck::cast_slice(logits.as_logical_slice::<f32>().unwrap()).to_vec(),
    }
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn f32_max_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    let a: &[f32] = bytemuck::cast_slice(a);
    let b: &[f32] = bytemuck::cast_slice(b);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Compare one arm with the reference; print every difference and return
/// whether the arm is bitwise equal.
fn compare(group: &str, split: &[usize], reference: &Capture, arm: &Capture, row_bytes: usize) -> bool {
    let label = format!("group={group} split={split:?}");
    let mut equal = true;
    let differing_rows: Vec<usize> = reference
        .rows
        .chunks_exact(row_bytes)
        .zip(arm.rows.chunks_exact(row_bytes))
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(row, _)| row)
        .collect();
    if !differing_rows.is_empty() {
        equal = false;
        let first = differing_rows[0];
        println!(
            "DS4_CHUNK_SPLIT_DIFF {label} final_state rows_differing={} first_row={first} last_row={} max_abs_first_row={:e}",
            differing_rows.len(),
            differing_rows.last().unwrap(),
            f32_max_abs_diff(
                &reference.rows[first * row_bytes..(first + 1) * row_bytes],
                &arm.rows[first * row_bytes..(first + 1) * row_bytes],
            ),
        );
    }
    for (a, b) in reference.cache.iter().zip(&arm.cache) {
        assert_eq!(a.label, b.label);
        if let Some(byte) = first_difference(&a.bytes, &b.bytes) {
            equal = false;
            let rows_differing = a
                .bytes
                .chunks_exact(a.row_bytes)
                .zip(b.bytes.chunks_exact(b.row_bytes))
                .filter(|(x, y)| x != y)
                .count();
            println!(
                "DS4_CHUNK_SPLIT_DIFF {label} cache {} first_row={} rows_differing={rows_differing}",
                a.label,
                byte / a.row_bytes,
            );
        }
    }
    if reference.logits != arm.logits {
        equal = false;
        println!(
            "DS4_CHUNK_SPLIT_DIFF {label} logits max_abs={:e}",
            f32_max_abs_diff(&reference.logits, &arm.logits)
        );
    }
    println!("DS4_CHUNK_SPLIT {label} bitwise_equal={equal}");
    equal
}

#[test]
#[ignore = "model-free but GPU-heavy: writes a ~12 GB synthetic GGUF and runs real-shape prefill; run under the lane hold"]
fn ds4_matrix_prefill_is_bitwise_chunk_split_invariant() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let cfg = config();
    let started = std::time::Instant::now();
    let (_directory, gguf) = write_fixture(&cfg);
    println!(
        "DS4_CHUNK_SPLIT fixture written in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let row_bytes = (cfg.hyper_connection_count * cfg.hidden_size) as usize * 4;
    let mut rng = SplitMix(726);
    let tokens: Vec<u32> = (0..GROUP_B_ROWS)
        .map(|_| (rng.next() % u64::from(cfg.vocab_size)) as u32)
        .collect();

    use Expect::{Differ, Equal, Report};
    // (split, cohort width, expectation). Cohort chunk sizes follow the
    // engine's `2048 / width` sizing: width 2 -> 1024, 3 -> 640, 4 -> 512.
    let groups: [(&str, usize, Vec<(Vec<usize>, usize, Expect)>); 2] = [
        (
            "A",
            GROUP_A_ROWS,
            vec![
                (vec![2048], 1, Equal),
                (vec![512, 1536], 1, Equal),
                (vec![1024, 1024], 1, Equal),
                (vec![640, 640, 640, 128], 1, Equal),
                (vec![256; 8], 1, Equal),
                (vec![1000, 1015, 33], 1, Equal),
                (vec![1000, 33, 1015], 1, Equal),
                (vec![130, 640, 640, 638], 1, Equal),
                (vec![1024, 1024], 2, Equal),
                (vec![640, 640, 640, 128], 3, Equal),
                (vec![512; 4], 4, Equal),
                // Sensitivity control: the token-wise tail must be seen.
                (vec![2047, 1], 1, Differ),
                // Below the matrix minimum (mv_id <= 32 rows, Q8_0 mv <= 8).
                (vec![2016, 32], 1, Report),
                (vec![2040, 8], 1, Report),
                (vec![2046, 2], 1, Report),
            ],
        ),
        (
            "B",
            GROUP_B_ROWS,
            vec![
                (vec![2048, 2049, 512], 1, Equal),
                (vec![2048, 2049, 256, 256], 1, Equal),
                (vec![2048, 2049, 33, 479], 1, Equal),
                (vec![2048, 2049, 479, 33], 1, Equal),
                (vec![640, 640, 640, 640, 640, 640, 640, 129], 1, Equal),
                (vec![1000, 1015, 33, 2048, 513], 1, Equal),
                (vec![640, 640, 640, 640, 640, 640, 640, 129], 3, Equal),
                (vec![512, 512, 512, 512, 512, 512, 512, 512, 256, 257], 4, Equal),
                (vec![2048, 2049, 511, 1], 1, Differ),
                (vec![2048, 2049, 480, 32], 1, Report),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (group, rows, arms) in &groups {
        let tokens = &tokens[..*rows];
        let (reference_split, reference_width, _) = &arms[0];
        let reference = run_split(&mut model, tokens, reference_split, *reference_width);
        let non_finite = bytemuck::cast_slice::<u8, f32>(&reference.rows)
            .iter()
            .filter(|value| !value.is_finite())
            .count();
        let last_row: &[f32] = bytemuck::cast_slice(&reference.rows[(rows - 1) * row_bytes..]);
        println!(
            "DS4_CHUNK_SPLIT group={group} reference={reference_split:?} non_finite_state_values={non_finite} last_row_rms={:e}",
            (last_row.iter().map(|v| v * v).sum::<f32>() / last_row.len() as f32).sqrt()
        );
        assert_eq!(non_finite, 0, "synthetic weights must keep the state finite");
        // The reference split runs again last: run-to-run control.
        for (split, width, expect) in arms[1..].iter().chain(std::iter::once(&arms[0])) {
            if *expect == Equal {
                assert!(split.iter().all(|&rows| rows >= MIN_MATRIX_APPEND_TOKENS));
            }
            let arm = run_split(&mut model, tokens, split, *width);
            let label = format!("{group} width={width}");
            let equal = compare(&label, split, &reference, &arm, row_bytes);
            match expect {
                Equal if !equal => failures.push(format!("{label} {split:?} differs")),
                Differ if equal => failures.push(format!("{label} {split:?} control equal")),
                _ => {}
            }
        }
    }
    println!(
        "DS4_CHUNK_SPLIT done in {:.1} s failures={failures:?}",
        started.elapsed().as_secs_f64()
    );
    assert!(failures.is_empty(), "chunk-split variant arms: {failures:?}");
}

/// What one arm must show against its group's reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Expect {
    /// Matrix chunks of at least `MIN_MATRIX_APPEND_TOKENS` rows: bitwise equal.
    Equal,
    /// Sensitivity control (token-wise tail): must differ.
    Differ,
    /// Below the matrix minimum: recorded, not asserted.
    Report,
}

/// One lane of a cooperative arm: its own tokens (from `seed`), `prefix` rows
/// prefilled solo before the cohort, then one `chunk`-row cohort chunk.
#[derive(Clone, Copy, Debug)]
struct Lane {
    seed: u64,
    prefix: usize,
    chunk: usize,
}

fn lane_tokens(lane: Lane, vocab: u32) -> Vec<u32> {
    let mut rng = SplitMix(lane.seed);
    (0..lane.prefix + lane.chunk)
        .map(|_| (rng.next() % u64::from(vocab)) as u32)
        .collect()
}

/// A cache holding `lane`'s prefix, prefilled solo in matrix chunks of at most
/// 2048 rows (every prefix used here splits into chunks of at least 33 rows).
fn prefilled(model: &mut Deepseek4Model, lane: Lane, tokens: &[u32]) -> Deepseek4Cache {
    let mut cache = model.allocate_cache(CONTEXT).unwrap();
    let mut offset = 0;
    while offset < lane.prefix {
        let len = (lane.prefix - offset).min(GROUP_A_ROWS);
        assert!(len >= MIN_MATRIX_APPEND_TOKENS || offset == 0);
        model
            .forward_verifier_prefill(&tokens[offset..offset + len], &mut cache)
            .unwrap();
        offset += len;
    }
    cache
}

fn lane_capture(model: &mut Deepseek4Model, cache: &Deepseek4Cache, state: &MlxBuffer) -> Capture {
    let last = model.last_token_state(state).unwrap();
    let logits = model.forward_logits(&last).unwrap();
    Capture {
        rows: bytemuck::cast_slice(state.as_logical_slice::<f32>().unwrap()).to_vec(),
        cache: cache_bytes(cache),
        logits: bytemuck::cast_slice(logits.as_logical_slice::<f32>().unwrap()).to_vec(),
    }
}

fn digest(capture: &Capture) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&capture.rows);
    for buffer in &capture.cache {
        hasher.update(buffer.label.as_bytes());
        hasher.update(&buffer.bytes);
    }
    hasher.update(&capture.logits);
    format!("{:x}", hasher.finalize())
}

/// `lane`'s solo route: its prefix, then its chunk as one matrix transaction.
fn solo_lane(model: &mut Deepseek4Model, lane: Lane) -> Capture {
    let tokens = lane_tokens(lane, model.cfg.vocab_size);
    let mut cache = prefilled(model, lane, &tokens);
    let state = model
        .forward_verifier_prefill(&tokens[lane.prefix..], &mut cache)
        .unwrap();
    assert_eq!(cache.position(), lane.prefix + lane.chunk);
    lane_capture(model, &cache, &state)
}

/// `lanes` (in pack order) ingest their chunks as one cooperative prefill
/// after solo prefixes; one capture per lane, in pack order.
fn cohort_lanes(model: &mut Deepseek4Model, lanes: &[Lane]) -> Vec<Capture> {
    let tokens: Vec<Vec<u32>> = lanes
        .iter()
        .map(|&lane| lane_tokens(lane, model.cfg.vocab_size))
        .collect();
    let mut caches: Vec<Deepseek4Cache> = lanes
        .iter()
        .zip(&tokens)
        .map(|(&lane, tokens)| prefilled(model, lane, tokens))
        .collect();
    let batches: Vec<&[u32]> = lanes
        .iter()
        .zip(&tokens)
        .map(|(lane, tokens)| &tokens[lane.prefix..])
        .collect();
    let states = {
        let mut refs: Vec<&mut Deepseek4Cache> = caches.iter_mut().collect();
        model
            .forward_verifier_prefill_cohort(&batches, &mut refs)
            .unwrap_or_else(|error| panic!("cohort {lanes:?}: {error:#}"))
    };
    lanes
        .iter()
        .zip(&caches)
        .zip(&states)
        .map(|((lane, cache), state)| {
            assert_eq!(cache.position(), lane.prefix + lane.chunk);
            lane_capture(model, cache, state)
        })
        .collect()
}

/// Lanes at different cache positions and chunk lengths (aggregate 1751 rows):
/// lane 1 crosses the ratio-128 block at 384, lane 3 runs the ratio-4 indexer
/// over more than `index_top_k` (512) compressed rows.
const UNEQUAL_LANES: [Lane; 4] = [
    Lane { seed: 0x5101, prefix: 0, chunk: 512 },
    Lane { seed: 0x5102, prefix: 300, chunk: 333 },
    Lane { seed: 0x5103, prefix: 1111, chunk: 129 },
    Lane { seed: 0x5104, prefix: 2111, chunk: 777 },
];

/// Lanes at different cache positions and chunk lengths ingest one cohort
/// chunk each: every lane equals its solo route bitwise at every pack offset,
/// and the transaction repeats bitwise. Every lane's digest is printed
/// (`DS4_COHORT_DIGEST`) so a second process run can be compared line for
/// line.
#[test]
#[ignore = "model-free but GPU-heavy: writes a ~12 GB synthetic GGUF and runs real-shape prefill; run under the lane hold"]
fn ds4_cohort_unequal_lanes_equal_solo_and_repeat_bitwise() {
    const REPEATS: usize = 5;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let cfg = config();
    let (_directory, gguf) = write_fixture(&cfg);
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let row_bytes = (cfg.hyper_connection_count * cfg.hidden_size) as usize * 4;
    assert!(UNEQUAL_LANES[3].prefix / 4 > cfg.index_top_k as usize);

    let solo: Vec<Capture> = UNEQUAL_LANES
        .iter()
        .map(|&lane| solo_lane(&mut model, lane))
        .collect();
    for (index, capture) in solo.iter().enumerate() {
        println!("DS4_COHORT_DIGEST arm=solo lane={index} sha={}", digest(capture));
    }

    let mut failures = Vec::new();
    let mut check = |arm: &str, order: &[usize], captures: &[Capture]| -> Vec<String> {
        let mut digests = Vec::new();
        for (offset, (&lane, capture)) in order.iter().zip(captures).enumerate() {
            let label = format!("{arm} lane={lane} offset={offset}");
            if !compare(
                "cohort",
                &[UNEQUAL_LANES[lane].chunk],
                &solo[lane],
                capture,
                row_bytes,
            ) {
                failures.push(format!("{label} differs from solo"));
            }
            let digest = digest(capture);
            println!("DS4_COHORT_DIGEST arm={arm} lane={lane} offset={offset} sha={digest}");
            digests.push(digest);
        }
        digests
    };

    // Every lane at every pack offset.
    let orders: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 3, 0, 1]];
    let mut first = Vec::new();
    for order in orders {
        let lanes: Vec<Lane> = order.iter().map(|&lane| UNEQUAL_LANES[lane]).collect();
        let captures = cohort_lanes(&mut model, &lanes);
        let digests = check(&format!("cohort{order:?}"), &order, &captures);
        if first.is_empty() {
            first = digests;
        }
    }

    // Run-to-run.
    let order = [0, 1, 2, 3];
    let mut repeat_failures = Vec::new();
    for repeat in 0..REPEATS {
        let captures = cohort_lanes(&mut model, &UNEQUAL_LANES);
        if check(&format!("repeat{repeat}"), &order, &captures) != first {
            repeat_failures.push(format!("repeat {repeat} digests differ"));
        }
    }
    failures.extend(repeat_failures);
    println!("DS4_COHORT_EQUALITY failures={failures:?}");
    assert!(failures.is_empty(), "cohort arms: {failures:?}");
}

/// Wall milliseconds of one cohort transaction over `lanes` (prefixes
/// prefilled first, untimed), or of the same chunks run solo one after another
/// when `solo` is set.
fn timed_chunks(model: &mut Deepseek4Model, lanes: &[Lane], solo: bool) -> f64 {
    let tokens: Vec<Vec<u32>> = lanes
        .iter()
        .map(|&lane| lane_tokens(lane, model.cfg.vocab_size))
        .collect();
    let mut caches: Vec<Deepseek4Cache> = lanes
        .iter()
        .zip(&tokens)
        .map(|(&lane, tokens)| prefilled(model, lane, tokens))
        .collect();
    let started = std::time::Instant::now();
    if solo {
        for ((lane, tokens), cache) in lanes.iter().zip(&tokens).zip(caches.iter_mut()) {
            model
                .forward_verifier_prefill(&tokens[lane.prefix..], cache)
                .unwrap();
        }
    } else {
        let batches: Vec<&[u32]> = lanes
            .iter()
            .zip(&tokens)
            .map(|(lane, tokens)| &tokens[lane.prefix..])
            .collect();
        let mut refs: Vec<&mut Deepseek4Cache> = caches.iter_mut().collect();
        model
            .forward_verifier_prefill_cohort(&batches, &mut refs)
            .unwrap();
    }
    started.elapsed().as_secs_f64() * 1e3
}

/// Cohort timing microbench (model-free, five real-shape layers): wall time of
/// one cohort transaction, including every command buffer's commit and wait,
/// against the same chunks run solo one after another, at the S3 n4 run's
/// shapes. Prints the median of `HF2Q_DS4_COHORT_BENCH_REPEATS` (default 5)
/// timed runs after one warm-up; asserts nothing. Decision-grade numbers need
/// a quiet host.
#[test]
#[ignore = "model-free microbench: writes a ~12 GB synthetic GGUF; run alone in a quiet window"]
fn ds4_cohort_microbench() {
    let repeats = std::env::var("HF2Q_DS4_COHORT_BENCH_REPEATS")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(5);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let cfg = config();
    let (_directory, gguf) = write_fixture(&cfg);
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let lane = |seed, prefix, chunk| Lane {
        seed,
        prefix,
        chunk,
    };
    // (shape, lanes): the cold wave's two cohort chunks (4 x 512 from 0, then
    // 4 x 237 from 512), turn 2's 4 x 119 rows after 896 cached rows, and a
    // Mixed slice of 2 x 128 rows after 1000.
    let shapes: Vec<(&str, Vec<Lane>)> = vec![
        (
            "cold_4x512_at0",
            (0..4).map(|i| lane(0x6100 + i, 0, 512)).collect(),
        ),
        (
            "cold_4x237_at512",
            (0..4).map(|i| lane(0x6110 + i, 512, 237)).collect(),
        ),
        (
            "warm_4x119_at896",
            (0..4).map(|i| lane(0x6120 + i, 896, 119)).collect(),
        ),
        (
            "mixed_2x128_at1000",
            (0..2).map(|i| lane(0x6130 + i, 1000, 128)).collect(),
        ),
    ];
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    for (shape, lanes) in &shapes {
        for (arm, solo) in [("solo_sum", true), ("cohort", false)] {
            timed_chunks(&mut model, lanes, solo);
            let times: Vec<f64> = (0..repeats)
                .map(|_| timed_chunks(&mut model, lanes, solo))
                .collect();
            println!(
                "DS4_COHORT_BENCH shape={shape} arm={arm} layers={} median_ms={:.2} min_ms={:.2} max_ms={:.2}",
                cfg.num_hidden_layers,
                median(times.clone()),
                times.iter().copied().fold(f64::INFINITY, f64::min),
                times.iter().copied().fold(0.0, f64::max),
            );
        }
    }
}

/// Transient prefill scratch (`MlxBufferPool` holdings: every buffer the
/// transaction's pool generations allocated, kept for reuse until released)
/// of one transaction over `lanes`, run as a cohort or, with `solo`, as one
/// solo chunk per lane. Prefixes are installed and the pool released first.
fn transaction_scratch_bytes(model: &mut Deepseek4Model, lanes: &[Lane], solo: bool) -> usize {
    use super::forward_support::{prefill_scratch_stats, release_prefill_scratch};
    let tokens: Vec<Vec<u32>> = lanes
        .iter()
        .map(|&lane| lane_tokens(lane, model.cfg.vocab_size))
        .collect();
    let mut caches: Vec<Deepseek4Cache> = lanes
        .iter()
        .zip(&tokens)
        .map(|(&lane, tokens)| prefilled(model, lane, tokens))
        .collect();
    release_prefill_scratch();
    if solo {
        let mut peak = 0;
        for ((lane, tokens), cache) in lanes.iter().zip(&tokens).zip(caches.iter_mut()) {
            model
                .forward_verifier_prefill(&tokens[lane.prefix..], cache)
                .unwrap();
            peak = peak.max(prefill_scratch_stats().free_bytes);
            release_prefill_scratch();
        }
        return peak;
    }
    let batches: Vec<&[u32]> = lanes
        .iter()
        .zip(&tokens)
        .map(|(lane, tokens)| &tokens[lane.prefix..])
        .collect();
    let mut refs: Vec<&mut Deepseek4Cache> = caches.iter_mut().collect();
    model
        .forward_verifier_prefill_cohort(&batches, &mut refs)
        .unwrap();
    let bytes = prefill_scratch_stats().free_bytes;
    release_prefill_scratch();
    bytes
}

/// Peak transient prefill scratch of cohort transactions against solo chunks
/// at the same aggregate rows (1 x 2048 vs 3 x 640 vs 4 x 512, cold and after
/// 896 cached rows). Prints `DS4_COHORT_SCRATCH` lines and asserts that every
/// cohort stays at or below the solo 2,048-row chunk's peak (the serving
/// envelope a cohort may not exceed).
#[test]
#[ignore = "model-free but GPU-heavy: writes a ~12 GB synthetic GGUF and runs real-shape prefill; run under the lane hold"]
fn ds4_cohort_prefill_scratch_against_solo() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let cfg = config();
    let (_directory, gguf) = write_fixture(&cfg);
    let mut model = Deepseek4Model::load_from_gguf(&gguf).unwrap();
    let lane = |seed, prefix, chunk| Lane {
        seed,
        prefix,
        chunk,
    };
    let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
    for prefix in [0, 896] {
        let solo_2048 = transaction_scratch_bytes(&mut model, &[lane(0x8100, prefix, 2048)], true);
        println!("DS4_COHORT_SCRATCH s0={prefix} shape=solo_1x2048 bytes={solo_2048} gib={:.3}", gib(solo_2048));
        for (shape, lanes) in [
            ("3x640", (0..3).map(|i| lane(0x8110 + i, prefix, 640)).collect::<Vec<_>>()),
            ("4x512", (0..4).map(|i| lane(0x8120 + i, prefix, 512)).collect::<Vec<_>>()),
        ] {
            let solo = transaction_scratch_bytes(&mut model, &lanes, true);
            println!("DS4_COHORT_SCRATCH s0={prefix} shape={shape} arm=solo_per_lane bytes={solo} gib={:.3}", gib(solo));
            let cohort = transaction_scratch_bytes(&mut model, &lanes, false);
            println!(
                "DS4_COHORT_SCRATCH s0={prefix} shape={shape} arm=cohort bytes={cohort} gib={:.3} vs_solo_2048={:.3}",
                gib(cohort),
                cohort as f64 / solo_2048 as f64
            );
            assert!(
                cohort <= solo_2048,
                "s0={prefix} {shape} cohort holds {cohort} bytes of scratch, above the solo 2048-row chunk's {solo_2048}"
            );
        }
    }
}
