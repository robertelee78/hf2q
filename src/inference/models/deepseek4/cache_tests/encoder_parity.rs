//! Ignored synthetic postprojection control, not warmup/model policy proof.
use crate::inference::models::deepseek4 as model;
use anyhow::{ensure, Result};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice};
use model::cache::{allocate_layer_from_plan, initialize_layer_state};
use model::cache::{Deepseek4CachePlan, LayerCache, LayerCacheSpan};
use model::cache_window_write::{encode_window_write, set_omit_copy};
use model::compressor_tail::{encode_indexer_tail, encode_main_tail};
#[path = "encoder_budget.rs"]
mod budget;
#[path = "encoder_packet.rs"]
mod packet;
use packet::{bf16_words, fill_bf16, window_expected, Packet, CACHE_BYTES, TOTAL_BYTES};

fn roots(layer: &LayerCache) -> impl Iterator<Item = &MlxBuffer> {
    std::iter::once(&layer.attention_kv)
        .chain(layer.indexer_kv.iter())
        .chain(layer.main_kv_state.iter())
        .chain(layer.main_score_state.iter())
        .chain(layer.indexer_kv_state.iter())
        .chain(layer.indexer_score_state.iter())
}
fn initialize(layer: &mut LayerCache, index: usize, canary: u16) -> Result<()> {
    fill_bf16(&mut layer.attention_kv, canary)?;
    if let Some(buffer) = &mut layer.indexer_kv {
        fill_bf16(buffer, canary)?;
    }
    initialize_layer_state(layer, index)?;
    Ok(())
}
fn known_window(layer: &LayerCache) -> Result<()> {
    let words = bf16_words(&layer.window_kv)?;
    for row in 0..64 {
        for col in 0..512 {
            ensure!(
                words[row * 512 + col] == window_expected(row, col),
                "independent window oracle row{row} col{col}"
            );
        }
    }
    Ok(())
}
fn copied_rows(
    destination: &MlxBuffer,
    source: &MlxBuffer,
    width: usize,
    canary: u16,
) -> Result<()> {
    let dst = bf16_words(destination)?;
    let src = bf16_words(source)?;
    ensure!(
        dst[..16 * width] == src[..16 * width],
        "compressed source/destination copy mismatch"
    );
    for row in 0..16 {
        ensure!(
            src[row * width..(row + 1) * width]
                .iter()
                .any(|word| *word != canary),
            "unchanged compressor row{row}"
        );
    }
    Ok(())
}
fn boundaries(layer: &LayerCache, ratio: u32, canary: u16, packet: &Packet) -> Result<()> {
    known_window(layer)?;
    ensure!(
        bf16_words(&layer.window_kv)?[64 * 512..]
            .iter()
            .all(|v| *v == canary),
        "window tail modified"
    );
    if let Some(cache) = &layer.compressed_kv {
        let valid = if ratio == 4 { 16 } else { 0 };
        if valid > 0 {
            copied_rows(cache, &packet.roots[3], 512, canary)?;
        }
        ensure!(
            bf16_words(cache)?[valid * 512..]
                .iter()
                .all(|v| *v == canary),
            "compressed tail modified"
        );
    }
    if let Some(cache) = &layer.indexer_kv {
        copied_rows(cache, &packet.roots[9], 128, canary)?;
        ensure!(
            bf16_words(cache)?[16 * 128..].iter().all(|v| *v == canary),
            "indexer tail modified"
        );
    }
    Ok(())
}
fn semantic<'a>(layer: &'a LayerCache, ratio: u32) -> Result<Vec<&'a [u8]>> {
    let mut result = Vec::with_capacity(7);
    result.push(&layer.window_kv.as_slice::<u8>()?[..64 * 512 * 2]);
    if ratio == 4 {
        result.push(&layer.compressed_kv.as_ref().unwrap().as_slice::<u8>()?[..16 * 512 * 2]);
        result.push(&layer.indexer_kv.as_ref().unwrap().as_slice::<u8>()?[..16 * 128 * 2]);
    }
    for buffer in layer
        .main_kv_state
        .iter()
        .chain(layer.main_score_state.iter())
        .chain(layer.indexer_kv_state.iter())
        .chain(layer.indexer_score_state.iter())
    {
        result.push(buffer.as_slice::<u8>()?);
    }
    Ok(result)
}
fn state_reset(layer: &mut LayerCache, index: usize) -> Result<()> {
    initialize_layer_state(layer, index)?;
    for state in layer
        .main_kv_state
        .iter()
        .chain(layer.indexer_kv_state.iter())
    {
        ensure!(state.as_slice::<f32>()?.iter().all(|v| v.to_bits() == 0));
    }
    for state in layer
        .main_score_state
        .iter()
        .chain(layer.indexer_score_state.iter())
    {
        ensure!(state
            .as_slice::<f32>()?
            .iter()
            .all(|v| v.to_bits() == f32::NEG_INFINITY.to_bits()));
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn encode<'a>(
    exec: &'a GraphExecutor,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &model::Deepseek4Config,
    layer: &LayerCache,
    span: &LayerCacheSpan,
    ratio: u32,
    packet: &Packet,
    window_only: bool,
) -> Result<mlx_native::graph::GraphSession<'a>> {
    let mut session = exec.begin()?;
    if ratio == 0 {
        // Ratio0 is allocation/layout + copy primitive ONLY, not dense encoder.
        dispatch_kv_cache_copy(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &packet.roots[12],
            &layer.window_kv,
            0,
            512,
            64,
            128,
            true,
        )?;
    } else {
        encode_window_write(
            &mut session,
            registry,
            device,
            layer,
            &packet.roots[12],
            &[&packet.roots[12]],
            Some(span),
            None,
            64,
            128,
            512,
        )?;
        if !window_only {
            let main = packet.main(ratio)?;
            let main = main.borrowed();
            let positions = packet.positions(if ratio == 4 { 16 } else { 0 })?;
            let cache = layer.compressed_kv.as_ref().unwrap();
            encode_main_tail(
                &mut session,
                registry,
                device,
                cfg,
                ratio,
                64,
                0,
                0,
                if ratio == 4 { 16 } else { 0 },
                cache,
                cache.shape()[0],
                layer.main_kv_state.as_ref().unwrap(),
                layer.main_score_state.as_ref().unwrap(),
                &main,
                &positions,
                &packet.roots[14],
            )?;
            if ratio == 4 {
                let indexer = packet.indexer()?;
                let indexer = indexer.borrowed();
                let cache = layer.indexer_kv.as_ref().unwrap();
                encode_indexer_tail(
                    &mut session,
                    registry,
                    device,
                    cfg,
                    64,
                    0,
                    0,
                    16,
                    cache,
                    cache.shape()[0],
                    layer.indexer_kv_state.as_ref().unwrap(),
                    layer.indexer_score_state.as_ref().unwrap(),
                    &indexer,
                    &positions,
                    &packet.roots[14],
                )?;
            }
        }
    }
    Ok(session)
}
fn drain_pair(
    first: mlx_native::graph::GraphSession<'_>,
    second: mlx_native::graph::GraphSession<'_>,
    ratio: u32,
    negative: bool,
) -> Result<()> {
    // Both fallible encodes finished before either commit. All packet/cache
    // roots and registry outlive this pair. No CPU access between submissions.
    let first = first.commit();
    let second = second.commit();
    let b = second.wait_until_completed();
    let a = first.wait_until_completed();
    eprintln!(
        "Q5_CACHE_COMMANDS ratio={ratio} negative={negative} small_ok={} large_ok={}",
        a.is_ok(),
        b.is_ok()
    );
    // Collect BOTH results before any assertion/error return/readback.
    a?;
    b?;
    Ok(())
}
struct OmissionGuard;
impl Drop for OmissionGuard {
    fn drop(&mut self) {
        set_omit_copy(false);
    }
}

#[test]
#[ignore = "synthetic 46,063,488B requested Metal payload; external process-group watchdog required"]
fn actual_cache_write_capacity_parity_fresh64() -> Result<()> {
    for name in [
        "MLX_UNRETAINED_REFS",
        "MLX_PROFILE_DISPATCH",
        "MLX_PROFILE_CB",
        "MLX_ENCODE_TRACE",
        "MLX_PROFILE_BARRIERS",
        "HF2Q_AUTO_BARRIER",
        "HF2Q_FORCE_SERIAL_DISPATCH",
        "HF2Q_PIPELINE_TG_MULT_HINT",
        "MLX_METAL_CAPTURE",
        "METAL_CAPTURE_ENABLED",
    ] {
        ensure!(std::env::var_os(name).is_none(), "requires {name} absent");
    }
    let cfg = super::official_config();
    let plans = [
        Deepseek4CachePlan::for_context(&cfg, 128)?,
        Deepseek4CachePlan::for_context(&cfg, 131072)?,
    ];
    let spans = [
        plans[0].plan_prefill_at(0, 64, 0)?,
        plans[1].plan_prefill_at(0, 64, 0)?,
    ];
    ensure!(spans[0] == spans[1]);
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Q5_CACHE_CAPABILITY unavailable_or_unknown device={error}");
            return Ok(());
        }
    };
    let exec = GraphExecutor::new(device.clone());
    let mut registry = KernelRegistry::new();
    let mut packet = Packet::allocate(&device, &cfg)?;
    let mut reference = vec![0u8; 1_048_576].into_boxed_slice();
    budget::check(&cfg, &plans, &spans, &packet)?;
    let mut cache_bytes = 0u64;
    let mut cache_roots = 0usize;
    for (index, ratio) in [(0, 0u32), (2, 4), (3, 128)] {
        let mut layers = [
            allocate_layer_from_plan(&device, &plans[0].layers[index], false)?,
            allocate_layer_from_plan(&device, &plans[1].layers[index], false)?,
        ];
        for layer in &layers {
            for root in roots(layer) {
                cache_bytes += root.byte_len() as u64;
                cache_roots += 1;
                ensure!(cache_bytes <= CACHE_BYTES && cache_roots <= 20);
            }
        }
        for (arm, layer) in layers.iter_mut().enumerate() {
            initialize(layer, index, if arm == 0 { 0x4480 } else { 0xc480 })?;
        }
        packet.prepare()?;
        let first = encode(
            &exec,
            &mut registry,
            &device,
            &cfg,
            &layers[0],
            &spans[0].layers[index],
            ratio,
            &packet,
            false,
        )?;
        let second = encode(
            &exec,
            &mut registry,
            &device,
            &cfg,
            &layers[1],
            &spans[1].layers[index],
            ratio,
            &packet,
            false,
        )?;
        drain_pair(first, second, ratio, false)?;
        for (arm, layer) in layers.iter().enumerate() {
            boundaries(
                layer,
                ratio,
                if arm == 0 { 0x4480 } else { 0xc480 },
                &packet,
            )?;
            let parts = semantic(layer, ratio)?;
            let mut offset = 0;
            for part in parts {
                let end = offset + part.len();
                ensure!(end <= reference.len());
                if arm == 0 {
                    reference[offset..end].copy_from_slice(part);
                } else {
                    ensure!(
                        &reference[offset..end] == part,
                        "ratio{ratio} semantic bit mismatch"
                    );
                }
                offset = end;
            }
            eprintln!(
                "Q5_CACHE_ARM completed ratio={ratio} capacity={} semantic_bytes={offset} scope={}",
                plans[arm].context_length,
                if ratio == 0 {
                    "allocation_layout_copy_primitive"
                } else {
                    "production_postprojection_tails"
                }
            );
        }
        if ratio == 4 {
            // Actual native parameter rejection; this session is NEVER submitted.
            let main = packet.main(4)?;
            let main = main.borrowed();
            let mut rejected_session = exec.begin()?;
            let invalid = mlx_native::ops::deepseek_compressor::dispatch_deepseek_compressor(
                rejected_session.encoder_mut(),
                &mut registry,
                &device,
                main.kv,
                main.score,
                main.ape,
                main.norm,
                layers[0].main_kv_state.as_ref().unwrap(),
                layers[0].main_score_state.as_ref().unwrap(),
                main.output,
                main.output,
                &mlx_native::ops::deepseek_compressor::DeepSeekCompressorParams {
                    batch: 1,
                    seq_len: 64,
                    start_pos: 0,
                    ratio: 4,
                    head_dim: 512,
                    cache_len: 15,
                    epsilon: cfg.rms_norm_eps,
                    write_cache: 0,
                },
            );
            drop(rejected_session);
            ensure!(invalid.is_err(), "native accepted cache_len15 for16 groups");
            eprintln!("Q5_CACHE_NEGATIVE short_cache_rejected=true");
            // Same canary makes equally broken windows agree; independent known
            // integer-BF16 expected writes must STILL reject both broken arms.
            for layer in &mut layers {
                fill_bf16(&mut layer.attention_kv, 0x4480)?;
            }
            set_omit_copy(true);
            let guard = OmissionGuard;
            let a = encode(
                &exec,
                &mut registry,
                &device,
                &cfg,
                &layers[0],
                &spans[0].layers[index],
                ratio,
                &packet,
                true,
            )?;
            let b = encode(
                &exec,
                &mut registry,
                &device,
                &cfg,
                &layers[1],
                &spans[1].layers[index],
                ratio,
                &packet,
                true,
            )?;
            drop(guard);
            drain_pair(a, b, ratio, true)?;
            ensure!(bf16_words(&layers[0].window_kv)? == bf16_words(&layers[1].window_kv)?);
            ensure!(
                known_window(&layers[0]).is_err() && known_window(&layers[1]).is_err(),
                "common omission escaped independent oracle"
            );
            eprintln!("Q5_CACHE_NEGATIVE common_omission_rejected=true equal_broken_windows=true");
        }
        for layer in &mut layers {
            state_reset(layer, index)?;
        }
    }
    ensure!(cache_bytes == CACHE_BYTES && cache_roots == 20);
    let bookkeeping = budget::check(&cfg, &plans, &spans, &packet)?;
    ensure!(reference.len() + 65536 + bookkeeping + 256 + 1024 <= 2_163_968);
    eprintln!("Q5_CACHE_HOST scoped_bookkeeping_bytes={bookkeeping} scoped_reference_bytes={} opaque_metadata_known=false",reference.len());
    eprintln!("Q5_CACHE_CAPABILITY completed temp_roots=15 cache_roots={cache_roots} requested_payload_bytes={TOTAL_BYTES} model_loaded=false runtime_admission=false opaque_metadata_known=false");
    Ok(())
}
