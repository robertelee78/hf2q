//! Per-arch registry entries.  ADR-012 P8 ships exactly two:
//!
//! * [`qwen35`] — Qwen3.5/3.6 dense (`Qwen3_5ForCausalLM`).
//! * [`qwen35moe`] — Qwen3.5/3.6 MoE (`Qwen3_5MoeForCausalLM`).
//!
//! Future arches (Gemma4 parity, Ministral ADR-015, DeepSeek-V3 ADR-016)
//! land their own file in their own ADR — no placeholders here per
//! Decision 20.

pub mod qwen35;
pub mod qwen35moe;

use super::registry::ArchEntry;

/// Iterator over every registered arch entry.  Order is registration order.
pub fn all() -> impl Iterator<Item = &'static ArchEntry> {
    [qwen35::ENTRY, qwen35moe::ENTRY].into_iter()
}
