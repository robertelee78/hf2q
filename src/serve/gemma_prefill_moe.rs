//! ADR-063 Q4-GEMMA-PREFILL-MOE (P1): route of the Gemma 4 prefill MoE
//! down projection (design `docs/research/adr063-q4-gemma-prefill-moe-design.md`
//! §4.1).
//!
//! The down call either flattens its rows to `seq_len·top_k` tokens with
//! `top_k = 1` (`Flat`, the previous route) or passes the `[seq_len, top_k,
//! K]` activations to the slotted expert entry point (`Slotted`). Both read
//! the same activation and weight bytes, write the same output offsets and
//! list each expert's rows in the same order, so they are bitwise equal
//! (Q1 Class A). Slotted launches `top_k`× fewer expert-matmul threadgroups.
//!
//! `HF2Q_GEMMA_PREFILL_MOE_DOWN=flat|slotted` exists only for the
//! same-binary A/B; it is removed at promotion (design §4.5, §7.4).

use mlx_native::MM_ID_ROUTING_THRESHOLD;

pub(crate) const PREFILL_MOE_DOWN_ENV: &str = "HF2Q_GEMMA_PREFILL_MOE_DOWN";

/// Shape of the Gemma prefill MoE down call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmaPrefillMoeDown {
    /// `n_tokens = seq_len·top_k`, `top_k = 1` (the previous route).
    Flat,
    /// `n_tokens = seq_len`, `top_k = top_k` on the slotted entry point.
    Slotted,
}

const PREFILL_MOE_DOWN_DEFAULT: GemmaPrefillMoeDown = GemmaPrefillMoeDown::Slotted;

impl GemmaPrefillMoeDown {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Slotted => "slotted",
        }
    }
}

fn parse_switch(raw: Option<&str>) -> std::result::Result<GemmaPrefillMoeDown, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(PREFILL_MOE_DOWN_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "flat" => Ok(GemmaPrefillMoeDown::Flat),
        "slotted" => Ok(GemmaPrefillMoeDown::Slotted),
        _ => Err(format!(
            "{PREFILL_MOE_DOWN_ENV} must be flat or slotted, got {raw:?}"
        )),
    }
}

fn switch_value(raw: Option<&str>) -> GemmaPrefillMoeDown {
    parse_switch(raw).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = PREFILL_MOE_DOWN_DEFAULT.as_str(),
            "invalid Gemma prefill MoE down switch"
        );
        PREFILL_MOE_DOWN_DEFAULT
    })
}

/// `HF2Q_GEMMA_PREFILL_MOE_DOWN`, resolved once at model load.
pub(crate) fn prefill_moe_down_from_env() -> GemmaPrefillMoeDown {
    let raw = std::env::var(PREFILL_MOE_DOWN_ENV).ok();
    let down = switch_value(raw.as_deref());
    tracing::info!(route = down.as_str(), "Gemma prefill MoE down route");
    down
}

/// The value the runtime-controls surface publishes (the environment is
/// fixed for the process, so it equals the value every load resolved).
pub(crate) fn published_prefill_moe_down() -> &'static str {
    static VALUE: std::sync::LazyLock<GemmaPrefillMoeDown> = std::sync::LazyLock::new(|| {
        switch_value(std::env::var(PREFILL_MOE_DOWN_ENV).ok().as_deref())
    });
    VALUE.as_str()
}

/// Expert-ID matmul parameters of one down call:
/// `(n_tokens, top_k, slotted)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DownCall {
    pub(crate) n_tokens: u32,
    pub(crate) top_k: u32,
    pub(crate) slotted: bool,
}

/// Route rule of the down call (DeepSeek-V4's paired-prefill rule,
/// `deepseek4::ffn_forward::use_paired_expert_prefill`):
/// slotted only when the switch asks for it, the stack is block-quantized or
/// scalar GGUF (not affine), the rows take `mm_id`
/// (`seq_len > MM_ID_ROUTING_THRESHOLD`, where the slotted contract accepts
/// them) and no `HF2Q_MM_ID_ROUTING_THRESHOLD` override moves that boundary.
/// Every other case keeps the flattened call, so the `mm_id`/`mv_id` choice
/// is today's in every case (design §3.1 R3).
pub(crate) fn down_call(
    switch: GemmaPrefillMoeDown,
    seq_len: usize,
    top_k: usize,
    affine: bool,
    threshold_override_present: bool,
) -> DownCall {
    let slotted = switch == GemmaPrefillMoeDown::Slotted
        && !affine
        && !threshold_override_present
        && seq_len > MM_ID_ROUTING_THRESHOLD as usize
        // The mm_id route (and its map0 instantiations) exists only for
        // these top_k; any other top_k keeps today's flattened call.
        && matches!(top_k, 1 | 6 | 8);
    if slotted {
        DownCall {
            n_tokens: seq_len as u32,
            top_k: top_k as u32,
            slotted: true,
        }
    } else {
        DownCall {
            n_tokens: (seq_len * top_k) as u32,
            top_k: 1,
            slotted: false,
        }
    }
}

/// `HF2Q_MM_ID_ROUTING_THRESHOLD` presence, the check DeepSeek-V4 makes.
pub(crate) fn mm_id_threshold_override_present() -> bool {
    std::env::var_os("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some()
}

/// Expert entry point of one down call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DownEntry {
    /// `dispatch_expert_matmul_id_pooled` (flattened rows).
    Flat,
    /// `dispatch_expert_matmul_id_pooled_slotted`: builds its own schedule.
    Slotted,
    /// `dispatch_expert_matmul_id_pooled_slotted_reusing_schedule`: reuses
    /// the schedule the gate/up `map0` left in the scratch (design §4.3).
    SlottedReusingSchedule,
}

/// Entry point of a down call (design §4.3, P2b). The gate/up call built a
/// reusable schedule only when both expert stacks are block-quantized (a
/// scalar stack never writes the scratch) and the call is slotted (the same
/// `mm_id` rows, ids, `n_tokens` and `top_k` as gate/up). A session the
/// body rotated between gate/up and down (profiling pre-finishes, the
/// imatrix intercept) takes the self-scheduling slotted call, so a valid
/// run never meets a key mismatch; without rotation a reuse error is a bug
/// and propagates.
pub(crate) fn down_entry(
    call: DownCall,
    both_stacks_block_quantized: bool,
    session_rotated: bool,
) -> DownEntry {
    if !call.slotted {
        DownEntry::Flat
    } else if both_stacks_block_quantized && !session_rotated {
        DownEntry::SlottedReusingSchedule
    } else {
        DownEntry::Slotted
    }
}

/// The native expert-matmul tile schedule the process resolves
/// (`HF2Q_MM_ID_SCHEDULE`), for the runtime-controls surface.
pub(crate) fn published_mm_id_schedule() -> serde_json::Value {
    serde_json::to_value(mlx_native::ggml_routing_policy_from_environment().expert_mm_schedule)
        .unwrap_or(serde_json::Value::Null)
}

/// Test-only record of every down entry point the prefill body chose on the
/// calling thread, in dispatch order (real-artifact gates assert reuse vs
/// fallback with it).
#[cfg(test)]
pub(crate) mod entry_probe {
    use super::DownEntry;
    use std::cell::RefCell;

    thread_local! {
        static ENTRIES: RefCell<Vec<DownEntry>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn record(entry: DownEntry) {
        ENTRIES.with(|entries| entries.borrow_mut().push(entry));
    }

    /// Every entry since the previous `take`.
    pub(crate) fn take() -> Vec<DownEntry> {
        ENTRIES.with(|entries| std::mem::take(&mut *entries.borrow_mut()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use GemmaPrefillMoeDown::{Flat, Slotted};

    #[test]
    fn gemma_prefill_moe_down_switch() {
        assert_eq!(parse_switch(None), Ok(Slotted));
        assert_eq!(parse_switch(Some("")), Ok(Slotted));
        assert_eq!(parse_switch(Some(" flat ")), Ok(Flat));
        assert_eq!(parse_switch(Some("FLAT")), Ok(Flat));
        assert_eq!(parse_switch(Some("slotted")), Ok(Slotted));
        assert!(parse_switch(Some("grid")).is_err());
        assert!(parse_switch(Some("0")).is_err());
        for invalid in ["grid", "0", "1", "on"] {
            assert_eq!(switch_value(Some(invalid)), PREFILL_MOE_DOWN_DEFAULT);
        }
        assert_eq!(Flat.as_str(), "flat");
        assert_eq!(Slotted.as_str(), "slotted");
        assert!(["flat", "slotted"].contains(&published_prefill_moe_down()));
    }

    #[test]
    fn gemma_prefill_moe_down_route_selection() {
        let threshold = MM_ID_ROUTING_THRESHOLD as usize;
        assert_eq!(threshold, 32, "design R3 pins the mm_id boundary");
        let flat = |seq_len: usize| DownCall {
            n_tokens: (seq_len * 8) as u32,
            top_k: 1,
            slotted: false,
        };
        for seq_len in [1, 5, 32] {
            assert_eq!(down_call(Slotted, seq_len, 8, false, false), flat(seq_len));
        }
        for seq_len in [33, 1420] {
            assert_eq!(
                down_call(Slotted, seq_len, 8, false, false),
                DownCall {
                    n_tokens: seq_len as u32,
                    top_k: 8,
                    slotted: true
                }
            );
        }
        for seq_len in [5, 32, 33, 1420] {
            // A threshold override can send gate/up to mv_id, where the
            // slotted contract refuses: every size keeps the flattened call.
            assert_eq!(down_call(Slotted, seq_len, 8, false, true), flat(seq_len));
            // Affine stacks and the A arm keep the flattened call.
            assert_eq!(down_call(Slotted, seq_len, 8, true, false), flat(seq_len));
            assert_eq!(down_call(Flat, seq_len, 8, false, false), flat(seq_len));
        }
        for top_k in [1usize, 6, 8] {
            assert!(down_call(Slotted, 232, top_k, false, false).slotted, "top_k {top_k}");
        }
        for top_k in [2usize, 4, 7, 16] {
            assert_eq!(
                down_call(Slotted, 232, top_k, false, false),
                DownCall {
                    n_tokens: (232 * top_k) as u32,
                    top_k: 1,
                    slotted: false
                },
                "top_k {top_k} has no mm_id route: flattened call"
            );
        }
    }

    #[test]
    fn gemma_prefill_moe_down_schedule_reuse_falls_back_on_rotation() {
        let slotted = down_call(Slotted, 232, 8, false, false);
        let flat = down_call(Flat, 232, 8, false, false);
        // Not rotated, block-quantized stacks: reuse the gate/up schedule.
        assert_eq!(
            down_entry(slotted, true, false),
            DownEntry::SlottedReusingSchedule
        );
        // Rotated (profile buckets, imatrix): self-scheduling slotted call.
        assert_eq!(down_entry(slotted, true, true), DownEntry::Slotted);
        // A scalar stack wrote no schedule.
        assert_eq!(down_entry(slotted, false, false), DownEntry::Slotted);
        for rotated in [false, true] {
            assert_eq!(down_entry(flat, true, rotated), DownEntry::Flat);
        }
        assert!(matches!(
            published_mm_id_schedule().as_str(),
            Some("uniform_grid" | "compact_tiles")
        ));
    }
}
