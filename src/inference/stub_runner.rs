//! Stub inference runner — returns UnsupportedPlatform on non-Apple targets.

use crate::inference::InferenceRunner;

/// Stub runner for platforms without MLX support.
pub struct StubRunner;

impl InferenceRunner for StubRunner {
    fn name(&self) -> &str {
        "stub"
    }

    fn is_available(&self) -> bool {
        false
    }
}
