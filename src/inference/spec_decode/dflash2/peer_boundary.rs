//! Test-only observations of real operands. This module does not exist in production.
use super::{config::DFlash2Config, staged::staged_tile_takes,
    tensors::{DFlash2LayerTensors, DFlash2Projection}, workspace::DFlash2Workspace};
use anyhow::{ensure, Context, Result};
use half::bf16;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const COPY: &str = "hf2q_peer_boundary_copy_u16";
const COPY_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void hf2q_peer_boundary_copy_u16(device const ushort *src [[buffer(0)]],
 device ushort *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
 uint gid [[thread_position_in_grid]]) { if (gid < count) dst[gid] = src[gid]; }
"#;

fn allocate(device: &MlxDevice, dtype: DType, shape: &[usize]) -> Result<MlxBuffer> {
    Ok(device.alloc_buffer(shape.iter().product::<usize>() * dtype.size_of(), dtype, shape.to_vec())?)
}

fn copy(encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    src: &MlxBuffer, dst: &MlxBuffer) -> Result<()> {
    use mlx_native::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};
    use mlx_native::metal::MTLSize;
    ensure!(src.byte_len() >= dst.byte_len() && dst.byte_len() % 2 == 0, "invalid raw copy extent");
    registry.register_source(COPY, COPY_SOURCE);
    let pipeline = registry.get_pipeline(COPY, device.metal_device())?;
    let count = u32::try_from(dst.byte_len() / 2)?;
    encode_with_args(encoder, pipeline, &[(0, KernelArg::Buffer(src)),
        (1, KernelArg::Buffer(dst)), (2, KernelArg::Bytes(as_bytes(&count)))],
        MTLSize::new(count as u64, 1, 1), MTLSize::new(256.min(count as u64), 1, 1));
    encoder.memory_barrier();
    Ok(())
}

struct ProjectionCapture {
    input: MlxBuffer,
    output: MlxBuffer,
    n: usize,
    k: usize,
    staged: bool,
}

impl ProjectionCapture {
    fn allocate(device: &MlxDevice, workspace: &DFlash2Workspace, n: usize, k: usize,
        bf16_input: bool) -> Result<Self> {
        Ok(Self { input: allocate(device, if bf16_input { DType::BF16 } else { DType::F32 }, &[8, k])?,
            output: allocate(device, DType::F32, &[8, n])?, n, k,
            staged: workspace.staged.is_some() && staged_tile_takes(n) })
    }
}

pub(super) struct BoundaryCapture {
    q: MlxBuffer,
    k: MlxBuffer,
    v: MlxBuffer,
    ring_k: MlxBuffer,
    ring_v: MlxBuffer,
    attention_output: MlxBuffer,
    lengths: MlxBuffer,
    positions: MlxBuffer,
    packed: bool,
    projections: [ProjectionCapture; 2],
}

impl BoundaryCapture {
    pub(super) fn allocate(device: &MlxDevice, config: &DFlash2Config,
        workspace: &DFlash2Workspace, tensors: &DFlash2LayerTensors) -> Result<Self> {
        ensure!(workspace.max_lanes == 1 && config.block_size == 8
            && config.num_attention_heads == 32 && config.num_key_value_heads == 8
            && config.head_dim == 128 && config.sliding_window == 2048,
            "boundary packet requires the original short B1 geometry");
        ensure!(matches!(tensors.output_projection, DFlash2Projection::Af4G64 { .. })
            && matches!(tensors.mlp_dynamic, DFlash2Projection::Af4G64 { .. }),
            "boundary packet requires actual AF4/G64 BF16 metadata");
        let ring = workspace.staged.as_ref().and_then(|s| s.attention.as_ref()).is_some();
        let packed_len = config.sliding_window - 1 + config.block_size;
        Ok(Self {
            q: allocate(device, DType::BF16, &[1, 32, 8, 128])?,
            k: allocate(device, DType::BF16, &[1, 8, if ring { 8 } else { packed_len }, 128])?,
            v: allocate(device, DType::BF16, &[1, 8, if ring { 16 } else { packed_len }, 128])?,
            ring_k: allocate(device, DType::BF16, &[1, 8, 2048, 128])?,
            ring_v: allocate(device, DType::BF16, &[1, 8, 2048, 128])?,
            attention_output: allocate(device, if ring { DType::BF16 } else { DType::F32 }, &[8, 4096])?,
            lengths: allocate(device, DType::U32, &[1])?, positions: allocate(device, DType::U32, &[1])?,
            packed: !ring,
            projections: [ProjectionCapture::allocate(device, workspace, config.hidden_size, 4096,
                ring || (workspace.staged.is_some() && staged_tile_takes(config.hidden_size)))?,
                ProjectionCapture::allocate(device, workspace, config.dynamic_projection_width(), config.hidden_size,
                    workspace.staged.is_some() && staged_tile_takes(config.dynamic_projection_width()))?],
        })
    }

    pub(super) fn attention(&self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, _config: &DFlash2Config, workspace: &DFlash2Workspace, layer: usize) -> Result<()> {
        if let Some(s) = workspace.staged.as_ref().and_then(|s| s.attention.as_ref()) {
            for (src, dst) in [(&s.queries, &self.q), (&s.current_keys, &self.k),
                (&s.current_values, &self.v), (&s.output, &self.attention_output)] {
                copy(encoder, registry, device, src, dst)?;
            }
        } else {
            for (src, dst) in [(&workspace.packed_query, &self.q), (&workspace.packed_key, &self.k),
                (&workspace.packed_value, &self.v), (&workspace.attention, &self.attention_output)] {
                copy(encoder, registry, device, src, dst)?;
            }
        }
        for (src, dst) in [(&workspace.ring_keys[layer], &self.ring_k),
            (&workspace.ring_values[layer], &self.ring_v), (&workspace.cache_lengths, &self.lengths),
            (&workspace.write_positions, &self.positions)] { copy(encoder, registry, device, src, dst)?; }
        Ok(())
    }

    pub(super) fn projection(&self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, workspace: &DFlash2Workspace, mlp: bool) -> Result<()> {
        let p = &self.projections[usize::from(mlp)];
        let input = if !mlp && !self.packed {
            &workspace.staged.as_ref().context("ring staged scratch")?.attention.as_ref().context("ring scratch")?.output
        } else if p.staged {
            &workspace.staged.as_ref().context("staged scratch")?.projection_input
        } else if mlp { &workspace.normalized } else { &workspace.attention };
        copy(encoder, registry, device, input, &p.input)?;
        copy(encoder, registry, device, if mlp { &workspace.dynamic } else { &workspace.projected }, &p.output)
    }

    // Called only after the original layer command completes. Replay has its own
    // input, output, table, reduction and partial buffers; it cannot overwrite the consumer.
    fn replay(&self, device: &MlxDevice, registry: &mut KernelRegistry, workspace: &DFlash2Workspace,
        projection: &DFlash2Projection, index: usize) -> Result<(MlxBuffer, MlxBuffer)> {
        use mlx_native::ops::{dflash2_af4g64_staged::{dispatch_dflash2_af4g64_staged,
            dflash2_af4g64_staged_partial_elements},
            qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16};
        let p = &self.projections[index];
        let mut common = allocate(device, DType::BF16, &[8, p.k])?;
        if p.input.dtype() == DType::BF16 {
            common.as_mut_slice::<u16>()?.copy_from_slice(p.input.as_slice::<u16>()?);
        } else {
            for (dst, src) in common.as_mut_slice::<u16>()?.iter_mut().zip(p.input.as_slice::<f32>()?) {
                *dst = bf16::from_f32(*src).to_bits();
            }
        }
        let DFlash2Projection::Af4G64 { weights, metadata } = projection else { anyhow::bail!("not AF4/G64") };
        let output = allocate(device, DType::F32, &[8, p.n])?;
        let mut encoder = device.command_encoder()?;
        if p.staged {
            let partials = allocate(device, DType::F32,
                &[dflash2_af4g64_staged_partial_elements(8, p.n as u32, p.k as u32).max(1)])?;
            dispatch_dflash2_af4g64_staged(&mut encoder, registry, device, &common, weights, metadata,
                &partials, &output, 8, p.n as u32, p.k as u32)?;
            encoder.commit_and_wait()?;
        } else {
            let mut widened = allocate(device, DType::F32, &[8, p.k])?;
            for (dst, bits) in widened.as_mut_slice::<f32>()?.iter_mut().zip(common.as_slice::<u16>()?) {
                *dst = bf16::from_bits(*bits).to_f32();
            }
            let table = allocate(device, workspace.af4_table.dtype(), workspace.af4_table.shape())?;
            let sums = allocate(device, workspace.af4_sums.dtype(), workspace.af4_sums.shape())?;
            dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16(&mut encoder, registry, device.metal_device(),
                &widened, &table, &sums, weights, metadata, &output, 8, p.n as u32, p.k as u32)?;
            encoder.commit_and_wait()?;
        }
        Ok((common, output))
    }

    pub(super) fn export(&self, directory: &Path, device: &MlxDevice, registry: &mut KernelRegistry,
        workspace: &DFlash2Workspace, tensors: &DFlash2LayerTensors, identity: Value) -> Result<()> {
        std::fs::create_dir(directory).context("packet directory must be new")?;
        let mut attention = serde_json::Map::new();
        attention.insert("q".into(), write_buffer(directory, "attention-q.bin", &self.q)?);
        let packed_len = self.k.shape()[2];
        let current = |buffer: &MlxBuffer, rows: usize| -> Result<Vec<u8>> {
            let source = buffer.as_slice::<u16>()?;
            let mut out = vec![0_u16; 8 * rows * 128];
            for head in 0..8 { for row in 0..8 {
                out[(head * rows + row) * 128..(head * rows + row + 1) * 128]
                    .copy_from_slice(&source[(head * packed_len + row) * 128..(head * packed_len + row + 1) * 128]);
            }}
            Ok(out.iter().flat_map(|bits| bits.to_le_bytes()).collect())
        };
        attention.insert("current_k".into(), write_bytes(directory, "attention-current-k.bin", "BF16", &[1,8,8,128], &current(&self.k,8)?)?);
        // Ring V is padded to 16 by the actual preparation; preserve its padding.
        let v = if self.packed { current(&self.v,16)? } else { self.v.as_slice::<u8>()?.to_vec() };
        attention.insert("current_v".into(), write_bytes(directory, "attention-current-v.bin", "BF16", &[1,8,16,128], &v)?);
        for (name, buffer) in [("ring_k", &self.ring_k), ("ring_v", &self.ring_v), ("output", &self.attention_output)] {
            attention.insert(name.into(), write_buffer(directory, &format!("attention-{name}.bin"), buffer)?);
        }
        let mut projections = Vec::new();
        for (index, (name, projection)) in [("layers.0.self_attn.o_proj.weight", &tensors.output_projection),
            ("layers.0.mlp_conv.kernel_projection.weight", &tensors.mlp_dynamic)].into_iter().enumerate() {
            let p = &self.projections[index];
            let (common, replay) = self.replay(device, registry, workspace, projection, index)?;
            let planes = logical_planes(directory, index, projection, p.n, p.k)?;
            let route = if p.staged { "dispatch_dflash2_af4g64_staged" } else { "dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16" };
            projections.push(json!({"name":name,"rows":8,"n":p.n,"k":p.k,"format":"AF4/G64","metadata_dtype":"BF16",
                "planes":planes.0,"source_plane_hashes":planes.1,
                "encoded_route":{"api":route,"source":"src/inference/spec_decode/dflash2/forward.rs:project/project_rounded",
                    "pipeline_witness":"source-bound API branch only; no unwitnessed kernel inferred"},
                "buffers":{"actual_input":write_buffer(directory,&format!("p{index}-actual-input.bin"),&p.input)?,
                    "actual_output":write_buffer(directory,&format!("p{index}-actual-output.bin"),&p.output)?,
                    "common_bf16_input":write_buffer(directory,&format!("p{index}-common-input.bin"),&common)?,
                    "hf_common_bf16_output":write_buffer(directory,&format!("p{index}-common-output.bin"),&replay)?}}));
        }
        let packet = json!({"schema":"hf2q.peer-draft-boundary.v1","identity":identity,
            "attention":{"layout":"hf-ring-v1","batch":1,"query_heads":32,"kv_heads":8,"rows":8,"head_dim":128,"window":2048,
                "lengths":self.lengths.as_slice::<u32>()?,"write_positions":self.positions.as_slice::<u32>()?,
                "encoded_route":{"api":if self.packed {"dispatch_dflash2_pack_attention_bf16 + dispatch_flash_attn_prefill_bf16_d128"} else {"dispatch_dflash2_ring_attention"},
                    "source":"src/inference/spec_decode/dflash2/forward.rs:dispatch_attention",
                    "pipeline_witness":"source-bound API branch only; no unwitnessed kernel inferred"},
                "representation_conversion":if self.packed {"Exact head-major copy of prepared BF16 packed Q; first 8 prepared K/V rows per head; zero-pad V rows8..16; unchanged persistent ring bytes; actual consumer F32 output"} else {"Exact copies of prepared ring operands and BF16 row-major consumer output"},
                "buffers":attention},"projections":projections});
        // Commit the packet last. A partial directory never masquerades as a packet.
        write_new(&directory.join("packet.json"), &serde_json::to_vec_pretty(&packet)?)?;
        Ok(())
    }
}

fn hash(bytes: &[u8]) -> String { hex::encode(Sha256::digest(bytes)) }
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?; file.sync_all()?; Ok(())
}
fn write_bytes(directory: &Path, file: &str, dtype: &str, shape: &[usize], bytes: &[u8]) -> Result<Value> {
    ensure!(shape.iter().all(|x| *x > 0), "empty packet tensor");
    let width = if dtype == "BF16" {2} else {4};
    ensure!(bytes.len() == shape.iter().product::<usize>() * width, "packet extent mismatch");
    if dtype == "BF16" { ensure!(bytes.chunks_exact(2).all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()), "nonfinite BF16 packet"); }
    if dtype == "F32" { ensure!(bytes.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()), "nonfinite F32 packet"); }
    write_new(&directory.join(file), bytes)?;
    Ok(json!({"file":file,"dtype":dtype,"shape":shape,"bytes":bytes.len(),"sha256":hash(bytes)}))
}
fn write_buffer(directory: &Path, file: &str, buffer: &MlxBuffer) -> Result<Value> {
    let dtype = match buffer.dtype() { DType::BF16 => "BF16", DType::F32 => "F32", DType::U32 => "U32", _ => anyhow::bail!("unsupported packet dtype") };
    write_bytes(directory, file, dtype, buffer.shape(), buffer.as_slice::<u8>()?)
}

// Exact inverse of tensors::prepare_af4_g64_planes, NOT a quantizer.
fn logical_planes(directory: &Path, index: usize, projection: &DFlash2Projection, n: usize, k: usize) -> Result<(Value, Value)> {
    let DFlash2Projection::Af4G64 { weights, metadata } = projection else { anyhow::bail!("not AF4/G64") };
    let source = weights.as_slice::<u32>()?;
    let meta = metadata.as_slice::<u32>()?;
    ensure!(n % 64 == 0 && k % 64 == 0 && source.len() == n*k/8 && meta.len() == n*k/64, "AF4 plane shape differs");
    let mut codes = vec![0_u32; n*k/8]; let mut scales = vec![0_u16; n*k/64]; let mut biases = scales.clone();
    for row in 0..n { for group in 0..k/64 {
        let value = meta[(row/64*(k/64)+group)*64+row%64];
        scales[row*(k/64)+group] = value as u16; biases[row*(k/64)+group] = (value>>16) as u16;
        for half in 0..2 { let group32 = group*2+half;
            for chunk in 0..4 { let word = source[((row/64*(k/32)+group32)*64+row%64)*4+chunk];
                for pair in 0..4 { let first = group32*32+16*(pair>>1)+4*chunk+2*(pair&1);
                    for (column, shift) in [(first,4*pair),(first+1,16+4*pair)] {
                        codes[row*(k/8)+column/8] |= ((word>>shift)&15) << (4*(column%8));
                    }
                }
            }
        }
    }}
    let codes = codes.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let scales = scales.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let biases = biases.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Ok((json!({"codes":write_bytes(directory,&format!("p{index}-codes.bin"),"U32",&[n,k/8],&codes)?,
        "scales":write_bytes(directory,&format!("p{index}-scales.bin"),"BF16",&[n,k/64],&scales)?,
        "biases":write_bytes(directory,&format!("p{index}-biases.bin"),"BF16",&[n,k/64],&biases)?}),
        json!({"prepared_codes_sha256":hash(weights.as_slice::<u8>()?),"prepared_metadata_sha256":hash(metadata.as_slice::<u8>()?),
            "logical_codes_sha256":hash(&codes),"logical_scales_sha256":hash(&scales),"logical_biases_sha256":hash(&biases)})))
}

pub(super) fn identity(device: &MlxDevice, family: &str, hidden: usize, fixture: &[u8], manifest_path: &Path,
    manifest: &Value, config_sha256: &str, weights_sha256: &str, target_sha256: &str) -> Result<Value> {
    let path = std::env::var_os("HF2Q_PEER_BOUNDARY_SOURCE_MANIFEST").context("HF2Q_PEER_BOUNDARY_SOURCE_MANIFEST required for observation")?;
    let bytes = std::fs::read(path)?;
    let consumer: Value = serde_json::from_slice(&bytes)?;
    for field in ["consumer_source","consumer_binary","dependencies","device","compiler"] {
        ensure!(consumer.get(field).is_some_and(|v| !v.is_null() && v != ""), "missing source identity {field}");
    }
    let binary = std::fs::read(std::env::current_exe()?)?;
    let binary_sha256 = hash(&binary);
    ensure!(consumer["consumer_binary"]["sha256"].as_str() == Some(binary_sha256.as_str())
        && consumer["consumer_binary"]["bytes"].as_u64() == Some(binary.len() as u64),
        "parent binary identity differs from actual current_exe");
    let lock = include_str!("../../../../Cargo.lock");
    let native = lock.split("[[package]]").find(|entry| entry.lines().any(|line| line == "name = \"mlx-native\""))
        .context("compiled lock has no mlx-native package")?;
    let native_version = native.lines().find_map(|line| line.strip_prefix("version = \"").and_then(|v| v.strip_suffix('"')))
        .context("compiled lock has no native version")?;
    Ok(json!({"family":family,"case":"short","layer":0,"hidden_size":hidden,
        "fixture_sha256":hash(fixture),"fixture_manifest_sha256":hash(&std::fs::read(manifest_path)?),
        "fixture_source":manifest["hf2q_source"],"fixture_producer_sha256":manifest["producer_sha256"],
        "checkpoint_config_sha256":config_sha256,"checkpoint_weights_sha256":weights_sha256,"target_sha256":target_sha256,
        "consumer":consumer,"consumer_source_manifest_sha256":hash(&bytes),"actual_binary_sha256":binary_sha256,
        "actual_binary_bytes":binary.len(),"actual_metal_device_name":device.metal_device().name(),
        "compiled_lock_mlx_native_version":native_version,"compiled_lock_sha256":hash(lock.as_bytes()),
        "compiled_sources":{"forward_sha256":hash(include_bytes!("forward.rs")),"staged_sha256":hash(include_bytes!("staged.rs")),
            "observer_sha256":hash(include_bytes!("peer_boundary.rs"))},
        "identity_authority":"Consumer source/dependency/compiler manifest is parent-declared provenance, not established by storing it. Binary SHA/bytes are independently matched to current_exe; Metal device name is queried. Native version is the compiled Cargo.lock record (no native runtime version API); parent locked-build proof binds it to the linked library. Outer proof independently binds checkout and dependency sources."}))
}

/// Byte-level proof surface, including snapshots taken before original scratch
/// reuse. No tolerance, digest-only comparison, fixture numeric acceptance, or
/// altered forward arithmetic is substituted for exact equality here.
pub(super) fn consumer_bytes(workspace: &DFlash2Workspace, snapshots: &[MlxBuffer; 6]) -> Result<Vec<Vec<u8>>> {
    let mut output = Vec::new();
    for buffer in [&workspace.hidden[1], &workspace.query, &workspace.key, &workspace.value,
        &workspace.qkv, &workspace.attention, &workspace.normalized, &workspace.dynamic,
        &workspace.convolved, &workspace.projected, &workspace.residual,
        &workspace.cache_lengths, &workspace.write_positions]
        .into_iter().chain(snapshots.iter()).chain(workspace.ring_keys.iter()).chain(workspace.ring_values.iter()) {
        output.push(buffer.as_slice::<u8>()?.to_vec());
    }
    if let Some(staged) = workspace.staged.as_ref() {
        if let Some(attention) = staged.attention.as_ref() {
            for buffer in [&attention.queries, &attention.current_keys, &attention.current_values, &attention.output] {
                output.push(buffer.as_slice::<u8>()?.to_vec());
            }
        }
    }
    Ok(output)
}
