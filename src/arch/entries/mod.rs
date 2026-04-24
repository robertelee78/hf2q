//! Per-arch registry entries — ADR-012 Decision 20.
//!
//! Exactly two entries ship in ADR-012: qwen35 and qwen35moe. Gemma4
//! parity follow-up, Ministral (ADR-015), and DeepSeek-V3 (ADR-016)
//! each add their own file in their own ADR. Per mantra: populated-
//! stub is still a stub. No placeholder files.

pub mod qwen35;
pub mod qwen35moe;
