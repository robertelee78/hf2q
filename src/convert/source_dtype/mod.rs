//! Source-dtype dequantizers — load-side conversion of safetensors-on-
//! disk dtypes to the F32 buffer the convert orchestrator expects.
//!
//! `F32 / F16 / BF16` live in `core::mlx_safetensors_loader::read_floats_to_f32`
//! (they're elementwise and shared with the runtime loader). This
//! module hosts the per-format dequantizers that the runtime path does
//! NOT need: today, just `float8_e4m3fn` for MiniMax-M2.7 / DeepSeek-V3
//! / other FP8-quantized HF releases.
//!
//! Per ADR-033 Decision §"FP8 source-dtype auto-detect": when
//! `config.json::quantization_config.quant_method == "fp8"`, the
//! `HfModelSource` loader dispatches FP8 tensor bytes through
//! [`fp8::dequantize_fp8_block`] before handing the resulting F32
//! buffer to `ConvertOrchestrator`. Modules listed in
//! `quantization_config.modules_to_not_convert` are read as their
//! native F32/BF16 directly (no FP8 dequant needed).

pub mod fp8;
