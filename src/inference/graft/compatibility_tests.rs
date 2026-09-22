use super::*;

/// GGUF fixture writer: typed metadata pairs plus optional F32 tensors
/// (`(name, dims, values)`), written to disk for `GgufFile::open` and
/// `GraftBank::load` (both parsers must accept the same bytes — the
/// artifact is one file, not two formats).
enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
}

fn write_fixture_gguf(
    path: &std::path::Path,
    meta: &[(&str, Meta)],
    tensors: &[(&str, Vec<u64>, Vec<f32>)],
) {
    use std::io::Write;
    let mut out = Vec::new();
    out.write_all(b"GGUF").unwrap();
    out.write_all(&3u32.to_le_bytes()).unwrap();
    out.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
    out.write_all(&(meta.len() as u64).to_le_bytes()).unwrap();
    for (key, value) in meta {
        out.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        out.write_all(key.as_bytes()).unwrap();
        match value {
            Meta::U32(v) => {
                out.write_all(&4u32.to_le_bytes()).unwrap();
                out.write_all(&v.to_le_bytes()).unwrap();
            }
            Meta::F32(v) => {
                out.write_all(&6u32.to_le_bytes()).unwrap();
                out.write_all(&v.to_le_bytes()).unwrap();
            }
            Meta::Bool(v) => {
                out.write_all(&7u32.to_le_bytes()).unwrap();
                out.write_all(&[*v as u8]).unwrap();
            }
            Meta::Str(s) => {
                out.write_all(&8u32.to_le_bytes()).unwrap();
                out.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                out.write_all(s.as_bytes()).unwrap();
            }
        }
    }
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
    let pad = (32 - out.len() % 32) % 32;
    out.extend(std::iter::repeat(0u8).take(pad));
    for (_, _, values) in tensors {
        for v in values {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    std::fs::write(path, out).expect("write fixture gguf");
}

/// A 12-layer qwen35 model, full-attention interval 4 → layers 3, 7, 11;
/// 4 KV heads, head_dim 256, rope 1e7 / rotary 64.
fn qwen35_model_meta() -> Vec<(&'static str, Meta)> {
    vec![
        ("general.architecture", Meta::Str("qwen35")),
        ("general.name", Meta::Str("Qwen3.5-27B")),
        ("qwen35.block_count", Meta::U32(12)),
        ("qwen35.full_attention_interval", Meta::U32(4)),
        ("qwen35.attention.head_count_kv", Meta::U32(4)),
        ("qwen35.attention.key_length", Meta::U32(256)),
        ("qwen35.rope.freq_base", Meta::F32(1e7)),
        ("qwen35.rope.dimension_count", Meta::U32(64)),
    ]
}

fn tmp_path(tag: &str) -> std::path::PathBuf {
    // Unique per call: `line!()` here would expand to THIS line for every
    // caller (a same-file race between parallel tests — surfaced as a
    // truncated-GGUF parse flake). An atomic counter guarantees distinct
    // fixture paths.
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "graft_compat_{tag}_{}_{}.gguf",
        std::process::id(),
        n
    ))
}

/// A structurally-matching bank: 2 slots, 4 KV heads, head_dim 256
/// (geometry comes from tensor dims), rope 1e7 / rotary 64 / IMROPE.
fn matching_bank_meta() -> Vec<(&'static str, Meta)> {
    vec![
        ("graft.mode", Meta::Str("splice_prefix")),
        ("graft.spec_version", Meta::U32(1)),
        ("graft.hook_point", Meta::Str("full_attn_kv")),
        ("graft.kind", Meta::Str("direct_kv")),
        ("graft.n_slots", Meta::U32(2)),
        ("graft.rope_theta", Meta::F32(1e7)),
        ("graft.rotary_dim", Meta::U32(64)),
        ("graft.mrope_interleaved", Meta::Bool(true)),
    ]
}

/// Bank tensors for the given layers with the given geometry.
fn bank_tensors(layers: &[u32], n_slots: u64, heads: u64, head_dim: u64) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    layers
        .iter()
        .flat_map(|layer| {
            let dims = vec![n_slots, heads, head_dim];
            let values = vec![0.5f32; (n_slots * heads * head_dim) as usize];
            vec![
                (format!("graft.k.{layer}"), dims.clone(), values.clone()),
                (format!("graft.v.{layer}"), dims, values),
            ]
        })
        .collect()
}

/// Write tensors with dynamic names (the fixture writer takes &str keys;
/// tensors use owned Strings, so serialize through a second writer call
/// is avoided by mapping into a Vec of owned triples here).
fn write_bank(
    path: &std::path::Path,
    meta: &[(&str, Meta)],
    layers: &[u32],
    n_slots: u64,
    heads: u64,
    head_dim: u64,
) {
    use std::io::Write;
    let tensors = bank_tensors(layers, n_slots, heads, head_dim);
    let mut out = Vec::new();
    out.write_all(b"GGUF").unwrap();
    out.write_all(&3u32.to_le_bytes()).unwrap();
    out.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
    out.write_all(&(meta.len() as u64).to_le_bytes()).unwrap();
    for (key, value) in meta {
        out.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        out.write_all(key.as_bytes()).unwrap();
        match value {
            Meta::U32(v) => {
                out.write_all(&4u32.to_le_bytes()).unwrap();
                out.write_all(&v.to_le_bytes()).unwrap();
            }
            Meta::F32(v) => {
                out.write_all(&6u32.to_le_bytes()).unwrap();
                out.write_all(&v.to_le_bytes()).unwrap();
            }
            Meta::Bool(v) => {
                out.write_all(&7u32.to_le_bytes()).unwrap();
                out.write_all(&[*v as u8]).unwrap();
            }
            Meta::Str(s) => {
                out.write_all(&8u32.to_le_bytes()).unwrap();
                out.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                out.write_all(s.as_bytes()).unwrap();
            }
        }
    }
    let mut offset = 0u64;
    for (name, dims, values) in &tensors {
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
    let pad = (32 - out.len() % 32) % 32;
    out.extend(std::iter::repeat(0u8).take(pad));
    for (_, _, values) in &tensors {
        for v in values {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    std::fs::write(path, out).expect("write bank gguf");
}

#[test]
fn qwen35_shape_extracts_full_attention_layers() {
    let path = tmp_path("shape");
    write_fixture_gguf(&path, &qwen35_model_meta(), &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let shape = GraftModelShape::from_gguf(&gguf).unwrap();
    assert_eq!(shape.arch, "qwen35");
    assert_eq!(shape.full_attn_layers, vec![3, 7, 11]);
    assert_eq!(shape.n_kv_heads, 4);
    assert_eq!(shape.head_dim, 256);
    assert_eq!(shape.rope_theta, 1e7);
    assert_eq!(shape.rotary_dim, 64);
    assert!(shape.mrope_interleaved);
    std::fs::remove_file(&path).ok();
}

#[test]
fn all_linear_model_refuses_the_site() {
    let mut meta = qwen35_model_meta();
    meta.retain(|(k, _)| *k != "qwen35.full_attention_interval");
    meta.push(("qwen35.full_attention_interval", Meta::U32(0)));
    let path = tmp_path("alllinear");
    write_fixture_gguf(&path, &meta, &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let err = GraftModelShape::from_gguf(&gguf).unwrap_err();
    assert!(err.to_string().contains("no full-attention layers"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn unsupported_architecture_is_refused_with_a_named_error() {
    let mut meta = qwen35_model_meta();
    meta[0] = ("general.architecture", Meta::Str("gemma4"));
    let path = tmp_path("gemma4");
    write_fixture_gguf(&path, &meta, &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let err = GraftModelShape::from_gguf(&gguf).unwrap_err();
    assert!(err.to_string().contains("does not expose the full_attn_kv graft site"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn structurally_matching_bank_binds() {
    let path = tmp_path("shape");
    write_fixture_gguf(&path, &qwen35_model_meta(), &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let shape = GraftModelShape::from_gguf(&gguf).unwrap();
    let bank_path = tmp_path("bank_ok");
    write_bank(&bank_path, &matching_bank_meta(), &[3, 7, 11], 2, 4, 256);
    let bank = GraftBank::load(&bank_path).unwrap();
    validate_graft_bank_for_model(&bank, &shape).unwrap();
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&bank_path).ok();
}

#[test]
fn linear_attention_layer_in_bank_is_fatal_and_named() {
    let path = tmp_path("shape");
    write_fixture_gguf(&path, &qwen35_model_meta(), &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let shape = GraftModelShape::from_gguf(&gguf).unwrap();
    let bank_path = tmp_path("bank_linear");
    write_bank(&bank_path, &matching_bank_meta(), &[3, 4], 2, 4, 256);
    let bank = GraftBank::load(&bank_path).unwrap();
    let err = validate_graft_bank_for_model(&bank, &shape).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("non-full-attention layer(s) [4]"), "{message}");
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&bank_path).ok();
}

#[test]
fn geometry_and_rope_mismatches_are_each_fatal() {
    let path = tmp_path("shape");
    write_fixture_gguf(&path, &qwen35_model_meta(), &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let shape = GraftModelShape::from_gguf(&gguf).unwrap();

    // (label, meta override, tensor heads, tensor head_dim, needle)
    let mut theta_meta = matching_bank_meta();
    theta_meta.retain(|(k, _)| *k != "graft.rope_theta");
    theta_meta.push(("graft.rope_theta", Meta::F32(1e6)));

    let mut rotary_meta = matching_bank_meta();
    rotary_meta.retain(|(k, _)| *k != "graft.rotary_dim");
    rotary_meta.push(("graft.rotary_dim", Meta::U32(128)));

    let mut mrope_meta = matching_bank_meta();
    mrope_meta.retain(|(k, _)| *k != "graft.mrope_interleaved");
    mrope_meta.push(("graft.mrope_interleaved", Meta::Bool(false)));

    let mut pos_meta = matching_bank_meta();
    pos_meta.push(("graft.position_base", Meta::U32(8)));

    let cases: Vec<(&str, Vec<(&'static str, Meta)>, u64, u64, &str)> = vec![
        ("kv heads", matching_bank_meta(), 8, 256, "n_kv_heads"),
        ("head dim", matching_bank_meta(), 4, 128, "head_dim"),
        ("rope theta", theta_meta, 4, 256, "rope_theta"),
        ("rotary dim", rotary_meta, 4, 256, "rotary_dim"),
        ("mrope", mrope_meta, 4, 256, "mrope_interleaved"),
        ("position base", pos_meta, 4, 256, "position_base"),
    ];
    for (label, meta, heads, head_dim, needle) in cases {
        let bank_path = tmp_path("bank_bad");
        write_bank(&bank_path, &meta, &[3, 7, 11], 2, heads, head_dim);
        let bank = GraftBank::load(&bank_path).unwrap();
        let err = validate_graft_bank_for_model(&bank, &shape).unwrap_err();
        assert!(err.to_string().contains(needle), "{label}: {err}");
        std::fs::remove_file(&bank_path).ok();
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn partial_coverage_bank_is_fatal() {
    let path = tmp_path("shape");
    write_fixture_gguf(&path, &qwen35_model_meta(), &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let shape = GraftModelShape::from_gguf(&gguf).unwrap();
    // Covers [3,7] but the model's full-attention set is [3,7,11].
    let bank_path = tmp_path("bank_partial");
    write_bank(&bank_path, &matching_bank_meta(), &[3, 7], 2, 4, 256);
    let bank = GraftBank::load(&bank_path).unwrap();
    let err = validate_graft_bank_for_model(&bank, &shape).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("does not cover full-attention layer(s) [11]"), "{message}");
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&bank_path).ok();
}

#[test]
fn zero_slot_canary_binds_on_supported_families_without_geometry_checks() {
    let path = tmp_path("shape");
    write_fixture_gguf(&path, &qwen35_model_meta(), &[]);
    let gguf = GgufFile::open(&path).unwrap();
    let shape = GraftModelShape::from_gguf(&gguf).unwrap();
    let mut meta = matching_bank_meta();
    meta.retain(|(k, _)| *k != "graft.n_slots");
    meta.push(("graft.n_slots", Meta::U32(0)));
    let bank_path = tmp_path("bank_canary");
    write_fixture_gguf(&bank_path, &meta, &[("token_embd.weight", vec![1], vec![0.0])]);
    let bank = GraftBank::load(&bank_path).unwrap();
    validate_graft_bank_for_model(&bank, &shape).unwrap();
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&bank_path).ok();
}

#[test]
fn loader_boundary_validates_identity_and_structure_together() {
    let model_path = tmp_path("model");
    write_fixture_gguf(&model_path, &qwen35_model_meta(), &[]);
    let model = GgufFile::open(&model_path).unwrap();

    // Matching name → Named compatibility (warn level, not fatal). The
    // Checkpoint level (declared 40-hex commit vs receipt-bound model
    // revision) is the same machinery GLP proves in its own tests.
    let mut meta = matching_bank_meta();
    meta.push(("general.base_model.0.name", Meta::Str("Qwen3.5-27B")));
    let named_path = tmp_path("bank_named");
    write_bank(&named_path, &meta, &[3, 7, 11], 2, 4, 256);
    let (_bank, compatibility) =
        validate_graft_for_model(&named_path, &model_path, &model).unwrap();
    assert_eq!(compatibility, Compatibility::Named);

    // Name mismatch → fatal at the loader boundary.
    let mut meta = matching_bank_meta();
    meta.push(("general.base_model.0.name", Meta::Str("Some-Unrelated-Model")));
    let mismatch_path = tmp_path("bank_mismatch");
    write_bank(&mismatch_path, &meta, &[3, 7, 11], 2, 4, 256);
    let err = validate_graft_for_model(&mismatch_path, &model_path, &model).unwrap_err();
    assert!(err.to_string().contains("base model mismatch"), "{err}");

    // Structural failure also surfaces through the loader boundary.
    let struct_path = tmp_path("bank_struct");
    write_bank(&struct_path, &matching_bank_meta(), &[4], 2, 4, 256);
    let err = validate_graft_for_model(&struct_path, &model_path, &model).unwrap_err();
    assert!(err.to_string().contains("non-full-attention"), "{err}");

    std::fs::remove_file(&model_path).ok();
    std::fs::remove_file(&named_path).ok();
    std::fs::remove_file(&mismatch_path).ok();
    std::fs::remove_file(&struct_path).ok();
}
