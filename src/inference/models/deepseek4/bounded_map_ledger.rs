//! Ignored metadata-only diagnostic. Native15 dependency must be normally pinned.
//! No Device, mmap, tensor payload access, or physical/admission authority.
use super::{
    weights::{required_tensor_specs, validate_tensor_catalog, TensorRole},
    Deepseek4Model,
};
use anyhow::{ensure, Context, Result};
use mlx_native::gguf::{GgufFile, GgufHeaderLimits, MetadataValue};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};
const MIB: u64 = 1024 * 1024;
const MAX_TENSORS: usize = 4096;
const OUTPUT_BYTES: usize = 2 * 1024 * 1024;
#[derive(Clone, Copy, Serialize)]
pub(super) struct Geometry {
    pub page_bytes: usize,
    pub max_buffer_bytes: usize,
}
impl Geometry {
    fn validate(self) -> Result<()> {
        ensure!(
            (4096..=65536).contains(&self.page_bytes) && self.page_bytes.is_power_of_two(),
            "invalid assumed host page geometry"
        );
        ensure!(
            self.max_buffer_bytes >= self.page_bytes
                && self.max_buffer_bytes <= 128 * 1024 * 1024 * 1024usize
                && self.max_buffer_bytes % self.page_bytes == 0,
            "invalid assumed max-buffer geometry"
        );
        Ok(())
    }
}
#[derive(Serialize, PartialEq, Eq, Debug)]
struct Identity {
    device: u64,
    inode: u64,
    bytes: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}
fn identity(file: &File) -> Result<Identity> {
    let m = file.metadata()?;
    ensure!(m.is_file(), "not a regular retained FD");
    Ok(Identity {
        device: m.dev(),
        inode: m.ino(),
        bytes: m.len(),
        mtime: (m.mtime(), m.mtime_nsec()),
        ctime: (m.ctime(), m.ctime_nsec()),
    })
}
fn open(path: &Path) -> Result<File> {
    ensure!(
        path.is_absolute() && path.as_os_str().as_encoded_bytes().len() <= 4096,
        "MODEL must be a bounded absolute path"
    );
    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    identity(&f)?;
    Ok(f)
}
fn limits() -> GgufHeaderLimits {
    GgufHeaderLimits {
        max_read_bytes: 64 * MIB,
        max_declared_allocation_bytes: 64 * MIB,
        max_metadata_entries: 4096,
        max_tensors: MAX_TENSORS as u64,
        max_string_bytes: MIB,
        max_total_array_elements: 1_000_000,
        max_nesting_depth: 4,
    }
}
#[derive(Serialize)]
struct Segment {
    logical_file: (usize, usize),
    mmap_file: (usize, usize),
    registered_file: (usize, usize),
    root_bytes: usize,
    mmap_bytes: usize,
    prefix_bytes: usize,
    tail_padding_bytes: usize,
}
#[derive(Serialize)]
struct Tensor<'a> {
    name: &'a str,
    shape: &'a [usize],
    ggml_type: String,
    required_role: Option<&'static str>,
    file: (usize, usize),
    logical_bytes: usize,
    segment: usize,
    logical_view_offset: usize,
    root_view_offset: usize,
    production_shape_and_storage_valid: bool,
    anonymous_f32_bytes: u64,
}
#[derive(Serialize)]
struct Report<'a> {
    schema: &'static str,
    source_commit: Option<&'static str>,
    native_version: &'static str,
    pid: u32,
    file: Identity,
    geometry: Geometry,
    geometry_authority: &'static str,
    model_loaded: bool,
    payload_read: bool,
    device_created: bool,
    physical_credit_bytes: u64,
    admission_authority: &'static str,
    header_read_bytes: u64,
    header_declared_reservation_bytes: u64,
    header_array_elements: u64,
    header_depth: u32,
    cpu_extra_declared_ceiling_bytes: u64,
    cpu_extra_declared_reservation_bytes: u64,
    cpu_accounting_basis: &'static str,
    output_ceiling_bytes: usize,
    config_debug: String,
    context: usize,
    output_limit: usize,
    logical_slots: usize,
    physical_cache_owners: &'static str,
    cache_requested_bytes_per_owner: u64,
    bootstrap_cache_policy: &'static str,
    catalog_exact: bool,
    required_count: usize,
    all_count: usize,
    required_present_count: usize,
    missing_required: Vec<&'a str>,
    unused_extras: Vec<&'a str>,
    required_logical_file_bytes: u64,
    all_logical_file_bytes: u64,
    unused_logical_file_bytes: u64,
    anonymous_f32_requested_bytes: u64,
    largest_f32_source_read_bytes: u64,
    all_coalesced_root_requested_bytes: usize,
    strict_catalog_weight_requested_upper_bytes: Option<u64>,
    all_mmap_extent_sum_bytes: usize,
    registered_file_page_union_bytes: usize,
    maximum_logical_file_end: usize,
    maximum_registered_file_end: usize,
    segments: Vec<Segment>,
    tensors: Vec<Tensor<'a>>,
}
struct Output {
    bytes: Vec<u8>,
}
impl Write for Output {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(b.len())
            .is_none_or(|n| n > OUTPUT_BYTES)
        {
            return Err(std::io::Error::other("bounded ledger output exceeds2MiB"));
        }
        self.bytes.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn encode(v: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(OUTPUT_BYTES)?;
    let mut out = Output { bytes };
    serde_json::to_writer(&mut out, v)?;
    Ok(out.bytes)
}
// Conservative declared source-work model, not allocator/OS/physical RAM.
// Reserve before invoking native planner/catalog and constructing output DTOs.
fn cpu_reservation(tensors: usize) -> Result<u64> {
    ensure!(tensors <= MAX_TENSORS, "extra CPU tensor count cap");
    let fixed = OUTPUT_BYTES as u64 + 1024 * 1024 + 64 * 4096;
    let per = (MAX_TENSORS as u64)
        .checked_mul(2048)
        .context("CPU reservation overflow")?;
    let total = fixed.checked_add(per).context("CPU reservation overflow")?;
    ensure!(total <= 16 * MIB, "declared extra CPU reservation cap");
    Ok(total)
}
fn add(sum: &mut u64, n: u64) -> Result<()> {
    *sum = sum.checked_add(n).context("ledger byte sum overflow")?;
    Ok(())
}
/// Actual bounded entry used by BOTH ignored real-artifact diagnostic and controls.
pub(super) fn inspect(file: File, geometry: Geometry) -> Result<Vec<u8>> {
    geometry.validate()?;
    let before = identity(&file)?;
    let header = GgufFile::from_file_with_limits(file.try_clone()?, limits())?;
    ensure!(
        header.metadata_string("general.architecture") == Some("deepseek4"),
        "not actual DeepSeek4 header"
    );
    ensure!(
        header
            .metadata_u32("deepseek4.block_count")
            .is_some_and(|n| (1..=64).contains(&n)),
        "layer-count source cap"
    );
    for key in [
        "deepseek4.attention.compress_ratios",
        "deepseek4.swiglu_clamp_exp",
        "deepseek4.swiglu_clamp_shexp",
    ] {
        ensure!(
            matches!(header.metadata(key),Some(MetadataValue::Array(v)) if v.len()<=64),
            "config-array source cap"
        );
    }
    ensure!(
        header
            .metadata_u32("deepseek4.embedding_length")
            .is_some_and(|n| n > 0 && n <= u32::MAX / 4),
        "strict-config multiplication safety cap"
    );
    let cpu_reserved = cpu_reservation(header.tensor_count())?;
    let cfg = Deepseek4Model::load_config_only(&header)?;
    let specs = required_tensor_specs(&cfg);
    ensure!(specs.len() <= MAX_TENSORS, "required catalog source cap");
    let roles: BTreeMap<_, _> = specs.iter().map(|s| (s.name.as_str(), s)).collect();
    let exact = validate_tensor_catalog(&header, &cfg).is_ok();
    let plan = header
        .mapped_tensor_storage_plan_with_limits(geometry.page_bytes, geometry.max_buffer_bytes)?;
    ensure!(
        plan.tensor_views().len() <= MAX_TENSORS && plan.segments().len() <= MAX_TENSORS,
        "pure map plan source cap"
    );
    let missing: Vec<_> = specs
        .iter()
        .filter(|s| header.tensor_info(&s.name).is_none())
        .map(|s| s.name.as_str())
        .collect();
    let mut extras = Vec::new();
    let mut tensors = Vec::new();
    tensors.try_reserve_exact(plan.tensor_views().len())?;
    let (mut all_bytes, mut required_bytes, mut extra_bytes, mut anon, mut host_peak) =
        (0, 0, 0, 0, 0);
    let mut present = 0;
    let mut all_valid = true;
    let mut max_logical = 0;
    for view in plan.tensor_views() {
        let info = header
            .tensor_info(view.name())
            .context("native plan view missing actual directory entry")?;
        ensure!(
            info.shape.len() <= 8 && view.file_end() as u64 <= before.bytes,
            "tensor metadata/file extent cap"
        );
        max_logical = max_logical.max(view.file_end());
        add(&mut all_bytes, info.byte_len as u64)?;
        let spec = roles.get(view.name());
        let mut f32_bytes = 0;
        let mut valid = false;
        let role = if let Some(spec) = spec {
            present += 1;
            add(&mut required_bytes, info.byte_len as u64)?;
            valid = info.shape == spec.shape
                && super::residency::metadata_storage_valid(spec, info.ggml_type);
            all_valid &= valid;
            Some(match spec.role {
                TensorRole::RawMatrix => "RawMatrix",
                TensorRole::IntegerLookupI32 => "IntegerLookupI32",
                TensorRole::ElementwiseF32 => {
                    f32_bytes = info
                        .shape
                        .iter()
                        .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                        .and_then(|n| n.checked_mul(4))
                        .context("F32 expansion overflow")?;
                    add(&mut anon, f32_bytes)?;
                    host_peak = host_peak.max(info.byte_len as u64);
                    "ElementwiseF32"
                }
            })
        } else {
            extras.push(view.name());
            add(&mut extra_bytes, info.byte_len as u64)?;
            None
        };
        tensors.push(Tensor {
            name: view.name(),
            shape: &info.shape,
            ggml_type: format!("{:?}", info.ggml_type),
            required_role: role,
            file: (view.file_start(), view.file_end()),
            logical_bytes: view.logical_bytes(),
            segment: view.segment_index(),
            logical_view_offset: view.logical_view_offset(),
            root_view_offset: view.root_byte_offset(),
            production_shape_and_storage_valid: valid,
            anonymous_f32_bytes: f32_bytes,
        });
    }
    let mut mmap_sum = 0usize;
    let mut max_registered = 0;
    let segments = plan
        .segments()
        .iter()
        .map(|s| -> Result<Segment> {
            ensure!(
                s.mapping_file_end() as u64 <= before.bytes,
                "mmap payload extent outside retained FD"
            );
            mmap_sum = mmap_sum
                .checked_add(s.mapping_bytes())
                .context("mmap sum overflow")?;
            max_registered = max_registered.max(s.registered_file_end());
            Ok(Segment {
                logical_file: (s.file_start(), s.file_end()),
                mmap_file: (s.mapping_file_start(), s.mapping_file_end()),
                registered_file: (s.registered_file_start(), s.registered_file_end()),
                root_bytes: s.root_byte_len(),
                mmap_bytes: s.mapping_bytes(),
                prefix_bytes: s.prefix_bytes(),
                tail_padding_bytes: s.tail_padding_bytes(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let usage = header
        .bounded_header_usage()
        .context("native bounded parser usage unavailable")?;
    let cache = super::cache::Deepseek4CachePlan::for_context(&cfg, 262_144)?;
    ensure!(
        identity(&file)? == before,
        "SAMEFD metadata changed during pure ledger"
    );
    let report = Report {
        schema: "hf2q.q5.deepseek.bounded-map-ledger.v1",
        source_commit: option_env!("HF2Q_BUILD_GIT_SHA"),
        native_version: env!("HF2Q_MLX_NATIVE_VERSION"),
        pid: std::process::id(),
        file: before,
        geometry,
        geometry_authority: "UnverifiedGeometry_CallerAssumed_NotDeviceCapability",
        model_loaded: false,
        payload_read: false,
        device_created: false,
        physical_credit_bytes: 0,
        admission_authority: "None_MetadataProjectionOnly",
        header_read_bytes: usage.read_bytes,
        header_declared_reservation_bytes: usage.declared_allocation_bytes,
        header_array_elements: usage.total_array_elements,
        header_depth: usage.maximum_nesting_depth,
        cpu_extra_declared_ceiling_bytes: 16 * MIB,
        cpu_extra_declared_reservation_bytes: cpu_reserved,
        cpu_accounting_basis: "DeclaredSourceWorkEnvelope_NotAllocatorOrPhysicalBound",
        output_ceiling_bytes: OUTPUT_BYTES,
        config_debug: format!("{cfg:?}"),
        context: 262_144,
        output_limit: 8192,
        logical_slots: 4,
        physical_cache_owners: "Unobserved_NoRuntimeOwnerClaim",
        cache_requested_bytes_per_owner: cache.resident_bytes,
        bootstrap_cache_policy:
            "PristineReuseFourNormalPlusC128Stub_ElseFiveNormal_NotSelectedHere",
        catalog_exact: exact && all_valid,
        required_count: specs.len(),
        all_count: header.tensor_count(),
        required_present_count: present,
        missing_required: missing,
        unused_extras: extras,
        required_logical_file_bytes: required_bytes,
        all_logical_file_bytes: all_bytes,
        unused_logical_file_bytes: extra_bytes,
        anonymous_f32_requested_bytes: anon,
        largest_f32_source_read_bytes: host_peak,
        all_coalesced_root_requested_bytes: plan.storage_plan().physical_byte_len(),
        strict_catalog_weight_requested_upper_bytes: if exact && all_valid {
            Some(
                (plan.storage_plan().physical_byte_len() as u64)
                    .checked_add(anon)
                    .context("weight requested sum overflow")?,
            )
        } else {
            None
        },
        all_mmap_extent_sum_bytes: mmap_sum,
        registered_file_page_union_bytes: plan.registered_file_page_union_bytes(),
        maximum_logical_file_end: max_logical,
        maximum_registered_file_end: max_registered,
        segments,
        tensors,
    };
    let out = encode(&report)?;
    ensure!(
        identity(&file)? == report.file,
        "SAMEFD changed while serializing ledger"
    );
    Ok(out)
}
#[test]
#[ignore = "root-supervised real DeepSeek SAMEFD bounded header/pure map plan only; no Device/model/payload"]
fn actual_deepseek_bounded_samefd_header_map_ledger() {
    let model = std::env::var_os("MODEL").expect("explicit root-owned MODEL path required");
    let geometry = Geometry {
        page_bytes: std::env::var("HF2Q_Q5_MAP_PAGE_BYTES")
            .expect("explicit assumed page size")
            .parse()
            .unwrap(),
        max_buffer_bytes: std::env::var("HF2Q_Q5_MAP_MAX_BUFFER_BYTES")
            .expect("explicit assumed max buffer")
            .parse()
            .unwrap(),
    };
    let before = (
        mlx_native::dispatch_count(),
        mlx_native::commit_count(),
        mlx_native::sync_count(),
    );
    let out = inspect(open(Path::new(&model)).unwrap(), geometry).unwrap();
    assert_eq!(
        before,
        (
            mlx_native::dispatch_count(),
            mlx_native::commit_count(),
            mlx_native::sync_count()
        )
    );
    println!(
        "Q5_DS_BOUNDED_MAP_LEDGER {}",
        std::str::from_utf8(&out).unwrap()
    );
}
#[cfg(test)]
#[path = "bounded_map_ledger_controls.rs"]
mod controls;
