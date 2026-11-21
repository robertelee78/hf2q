//! Native fresh-HTTP recipe metadata only: no model/device/cache/acquisition.
use super::{fresh_http, output_preflight, Route, Q36, Q38};
use crate::adr063_invariance::{
    acquisition::{self, FileBinding, FileIdentity},
    receipt::Family,
    store::hash,
};
use crate::inference::models::qwen35::{Qwen35Config, Qwen35Variant};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    family: Family,
    model: PathBuf,
    file_identity: FileIdentity,
    full_content_receipt: FileBinding,
    case: FileBinding,
    expected_exporter_binary_sha256: String,
    output: PathBuf,
}

#[derive(Serialize)]
struct Artifact<'a> {
    path: &'a Path,
    sha256: &'a str,
}

#[derive(Serialize)]
struct Export<'a> {
    schema: &'static str,
    family: Family,
    binary_sha256: &'a str,
    native_version: &'static str,
    compiled_source_stamp: Option<&'static str>,
    model: &'a Path,
    model_file: &'a FileIdentity,
    model_full_sha256_expected: &'static str,
    full_content_receipt: &'a FileBinding,
    case: &'a FileBinding,
    packet: Artifact<'a>,
    model_config: Artifact<'a>,
    recipe: Artifact<'a>,
    header_bytes: u64,
    header_read_bytes: u64,
    header_declared_allocation_bytes: u64,
    header_total_array_elements: u64,
    header_maximum_nesting_depth: u32,
    declared_header_allocation_is_physical_bound: bool,
    header_config_context_override: bool,
    model_payload_rehashed: bool,
    model_loaded: bool,
    device_created: bool,
    context_created: bool,
    inference_performed: bool,
    acquisition_attested: bool,
    whole_c2_pass: bool,
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn run() -> Result<()> {
    let path = PathBuf::from(std::env::var_os("HF2Q_C2_CONFIG_RECIPE_PACKET")
        .context("native config recipe packet missing")?);
    let input = OpenOptions::new().read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(&path)?;
    let info = input.metadata()?;
    ensure!(path.is_absolute() && info.is_file() && info.len() <= 4 * 1024 * 1024,
        "bounded regular absolute native recipe packet required");
    let mut raw = Vec::with_capacity(info.len() as usize);
    input.take(4 * 1024 * 1024 + 1).read_to_end(&mut raw)?;
    ensure!(raw.len() <= 4 * 1024 * 1024, "native recipe packet exceeds4MiB");
    let packet: Packet = serde_json::from_slice(&raw)?;
    ensure!(packet.schema == "hf2q.c2.native-config-recipe-packet.v1" && packet.model.is_absolute(),
        "native recipe schema and absolute model required");
    let binary = acquisition::executable_digest()?;
    ensure!(binary == packet.expected_exporter_binary_sha256, "actual recipe exporter binary differs");
    let (family_name, model_sha256) = match packet.family {
        Family::Qwen38 => ("qwen38", Q38),
        Family::Qwen36 => ("qwen36", Q36),
        _ => anyhow::bail!("native recipe requires an explicit Qwen family"),
    };
    let receipt = acquisition::checked_file(&packet.full_content_receipt, 16 * 1024 * 1024)?;
    acquisition::full_content_receipt_has_model(&receipt, model_sha256)?;
    let case: serde_json::Value = serde_json::from_slice(
        &acquisition::checked_file(&packet.case, 4 * 1024 * 1024)?)?;
    ensure!(case["schema"] == "hf2q.q5.calibration-case.v1" && case["family"] == family_name
        && case["workload"]["slots"] == 4 && case["workload"]["context_per_slot"] == 262144
        && case["workload"]["anchors_per_slot"] == 4, "exact original full four-slot K4 case required");
    let quota = case["workload"]["anchor_bytes_per_slot_max"].as_u64()
        .context("source-bound K4 anchor quota absent")?;
    ensure!(quota > 0 && quota % 5 == 0, "explicit five-snapshot K4 source quota required");
    output_preflight(&packet.output)?;
    let file = OpenOptions::new().read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(&packet.model)?;
    let before = FileIdentity::observed(&file)?;
    ensure!(before == packet.file_identity, "recipe model FD differs from bound identity");
    let header = mlx_native::gguf::GgufFile::from_file_with_limits(
        file.try_clone()?,
        mlx_native::gguf::GgufHeaderLimits {
            max_read_bytes: 32 * 1024 * 1024,
            max_declared_allocation_bytes: 128 * 1024 * 1024,
            max_metadata_entries: 100_000,
            max_tensors: 100_000,
            max_string_bytes: 32 * 1024 * 1024,
            max_total_array_elements: 2_000_000,
            max_nesting_depth: 8,
        },
    )?;
    let usage = header.bounded_header_usage().context("bounded header counters absent")?;
    ensure!(header.tensor_data_offset() <= 32 * 1024 * 1024, "selected header exceeds32MiB");
    let config = Qwen35Config::from_gguf(&header)?;
    ensure!(matches!((packet.family, config.variant),
        (Family::Qwen38, Qwen35Variant::Dense) | (Family::Qwen36, Qwen35Variant::Moe)),
        "recipe architecture/family mismatch");
    let config_raw = serde_json::to_vec(&config)?;
    let route = format!("C2-fresh-fixed-HTTP-{:?};quota={quota}", Route::Off);
    let recipe = fresh_http::config_recipe(packet.family, &route, hash(&config_raw));
    let recipe_raw = serde_json::to_vec(&recipe)?;
    let recipe_sha256 = hash(&recipe_raw);
    ensure!(FileIdentity::observed(&file)? == before, "model FD changed during header export");
    fs::create_dir(&packet.output)?;
    fs::set_permissions(&packet.output, fs::Permissions::from_mode(0o700))?;
    let config_path = packet.output.join("model-config.json");
    let recipe_path = packet.output.join("recipe.json");
    write_new(&config_path, &config_raw)?;
    write_new(&recipe_path, &recipe_raw)?;
    let packet_sha256 = hash(&raw);
    let exported = Export {
        schema: "hf2q.c2.native-config-recipe-export.v1", family: packet.family,
        binary_sha256: &binary, native_version: env!("HF2Q_MLX_NATIVE_VERSION"),
        compiled_source_stamp: option_env!("HF2Q_BUILD_GIT_SHA"), model: &packet.model,
        model_file: &before, model_full_sha256_expected: model_sha256,
        full_content_receipt: &packet.full_content_receipt, case: &packet.case,
        packet: Artifact { path: &path, sha256: &packet_sha256 },
        model_config: Artifact { path: &config_path, sha256: &recipe.actual_model_config_sha256 },
        recipe: Artifact { path: &recipe_path, sha256: &recipe_sha256 },
        header_bytes: header.tensor_data_offset(), header_read_bytes: usage.read_bytes,
        header_declared_allocation_bytes: usage.declared_allocation_bytes,
        header_total_array_elements: usage.total_array_elements,
        header_maximum_nesting_depth: usage.maximum_nesting_depth,
        declared_header_allocation_is_physical_bound: false,
        header_config_context_override: false,
        model_payload_rehashed: false, model_loaded: false, device_created: false,
        context_created: false, inference_performed: false, acquisition_attested: false,
        whole_c2_pass: false,
    };
    write_new(&packet.output.join("export.json"), &serde_json::to_vec_pretty(&exported)?)?;
    println!("C2_NATIVE_CONFIG_RECIPE_EXPORTED family={:?} config_sha256={} model=false device=false whole_c2=false",
        packet.family, recipe_sha256);
    Ok(())
}

#[test]
#[ignore = "bound header/JSON recipe export only; no model/device/context/inference"]
fn qwen_c2_native_config_recipe_export() {
    run().unwrap();
}
