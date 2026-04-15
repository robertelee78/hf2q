//! GPU-related utilities.
//!
//! ADR-008: the candle-based device selection, tensor conversion, and transformer
//! forward pass have been removed.  Inference uses mlx-native exclusively.
//! This module retains only the tokenizer utilities needed by calibration paths.

pub mod tokenizer;
