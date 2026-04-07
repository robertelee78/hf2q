//! Model architecture definitions and dispatch.
//!
//! This module provides:
//! - [`ModelArchitecture`] — enum of supported model architectures
//! - [`detect_architecture`] — reads config.json and determines the architecture
//! - [`Gemma4Model`] — Gemma 4 forward pass implementation
//! - Architecture validation and error reporting

#[cfg(feature = "mlx-native")]
pub mod registry;

#[cfg(feature = "mlx-native")]
pub mod gemma4;
