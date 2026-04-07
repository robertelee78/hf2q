//! Model architecture definitions and dispatch.
//!
//! This module provides:
//! - [`ModelArchitecture`] — enum of supported model architectures
//! - [`detect_architecture`] — reads config.json and determines the architecture
//! - Architecture validation and error reporting

#[cfg(feature = "mlx-native")]
pub mod registry;
