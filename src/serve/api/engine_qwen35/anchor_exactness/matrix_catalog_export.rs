//! Compiled case/artifact obligations only; no model, device or acquisition.
use super::*;
use super::matrix_plan::{self, Mode};
use std::{collections::BTreeMap, fs::OpenOptions, io::{BufWriter, Write}, os::unix::fs::OpenOptionsExt};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    expected_exporter_binary_sha256: String,
    mode: Mode,
    graft: bool,
    output: PathBuf,
}
#[derive(Serialize)]
struct Catalog<'a> {
    schema: &'static str,
    actual_exporter_binary_sha256: &'a str,
    native_version: &'static str,
    mode: Mode,
    graft: bool,
    full_case_catalog: &'a [String],
    catalog_sha256: String,
    required_artifacts: BTreeMap<&'a str, Vec<String>>,
    model_opened: bool,
    device_created: bool,
    acquisition_attested: bool,
}
fn run() -> Result<()> {
    let path = PathBuf::from(std::env::var_os("HF2Q_C2_MATRIX_CATALOG_PACKET")
        .context("compiled matrix catalog packet missing")?);
    let input = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(&path)?;
    ensure!(path.is_absolute() && input.metadata()?.is_file() && input.metadata()?.len() <= 65536,
        "bounded regular absolute catalog packet required");
    let mut raw = Vec::new();
    input.take(65537).read_to_end(&mut raw)?;
    ensure!(raw.len() <= 65536, "catalog packet exceeds64KiB");
    let packet: Packet = serde_json::from_slice(&raw)?;
    ensure!(packet.schema == "hf2q.c2.matrix-catalog-request.v1", "catalog request schema differs");
    let binary = acquisition::executable_digest()?;
    ensure!(binary == packet.expected_exporter_binary_sha256, "actual catalog exporter binary differs");
    output_preflight(&packet.output)?;
    let cases = matrix_plan::catalog(packet.mode, packet.graft);
    let required = cases.iter().map(|case| (case.as_str(), matrix_plan::required_artifacts(case))).collect();
    let catalog = Catalog { schema: "hf2q.c2.compiled-matrix-catalog.v1",
        actual_exporter_binary_sha256: &binary, native_version: env!("HF2Q_MLX_NATIVE_VERSION"),
        mode: packet.mode, graft: packet.graft, full_case_catalog: &cases,
        catalog_sha256: hash(&serde_json::to_vec(&cases)?), required_artifacts: required,
        model_opened: false, device_created: false, acquisition_attested: false };
    fs::create_dir(&packet.output)?;
    fs::set_permissions(&packet.output, fs::Permissions::from_mode(0o700))?;
    let file = OpenOptions::new().create_new(true).write(true).mode(0o600).open(packet.output.join("catalog.json"))?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer(&mut output, &catalog)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}
#[test]
#[ignore = "bound compiled whole-case/artifact catalog only; no model/device/acquisition"]
fn qwen_c2_matrix_catalog_export() { run().unwrap(); }
