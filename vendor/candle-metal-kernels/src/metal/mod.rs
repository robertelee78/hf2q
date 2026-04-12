pub mod buffer;
pub mod command_buffer;
pub mod commands;
pub mod compute_pipeline;
pub mod device;
pub mod encoder;
// hf2q ADR-006 Phase 0 vendor patch (2026-04-11): per-encoder GPU
// timestamp instrumentation. Opt-in via `HF2Q_PHASE0_INSTRUMENT=1` +
// `HF2Q_PHASE0_INSTRUMENT_DUMP=<path>`; uninstrumented builds pay for
// one atomic-bool load per encoder creation and nothing else.
pub mod instrument;
pub mod library;

pub use buffer::*;
pub use command_buffer::*;
pub use commands::*;
pub use compute_pipeline::*;
pub use device::*;
pub use encoder::*;
pub use library::*;
