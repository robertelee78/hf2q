//! ADR-065 E4 Qwen critical-path interval observer.
//!
//! `HF2Q_ADR065_CRITICAL_PATH=1` emits CLOCK_MONOTONIC intervals around
//! bounded prefill slices and complete joint/solo decode turns. Spans use
//! existing operation boundaries and add no synchronization.

use std::sync::LazyLock;

use serde_json::json;

use crate::serve::multi_seq_kv::SlotId;

pub(super) const ENV: &str = "HF2Q_ADR065_CRITICAL_PATH";
pub(super) const SCHEMA: &str = "hf2q.adr065.critical-path-interval.v1";
const PREFIX: &str = "[ADR065_CRITICAL_PATH_JSON] ";

static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var(ENV).as_deref() == Ok("1"));

pub(in crate::serve::api) fn enabled() -> bool {
    *ENABLED
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Lane {
    slot: u32,
    verify_eligibility: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Detail {
    Decode {
        mode: &'static str,
        lanes: Vec<Lane>,
    },
    Prefill {
        slot: u32,
        prompt_tokens: usize,
        chunk_start: usize,
        chunk_end: usize,
    },
}

pub(in crate::serve::api) struct Span {
    begin_ns: u64,
    detail: Detail,
}

impl Span {
    pub(in crate::serve::api) fn decode<'a>(
        mode: &'static str,
        lanes: impl IntoIterator<Item = (SlotId, &'a str)>,
    ) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let lanes = lanes
            .into_iter()
            .map(|(slot, status)| Lane {
                slot: slot.0,
                verify_eligibility: match status {
                    "eligible" => "eligible",
                    "blocked_transform" => "blocked_transform",
                    "no_draft" => "no_draft",
                    _ => "unknown",
                },
            })
            .collect::<Vec<_>>();
        let begin_ns = crate::serve::phase_trace::monotonic_ns().ok()?;
        Some(Self {
            begin_ns,
            detail: Detail::Decode { mode, lanes },
        })
    }

    pub(in crate::serve::api) fn prefill(
        slot: SlotId,
        prompt_tokens: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let begin_ns = crate::serve::phase_trace::monotonic_ns().ok()?;
        Some(Self {
            begin_ns,
            detail: Detail::Prefill {
                slot: slot.0,
                prompt_tokens,
                chunk_start,
                chunk_end,
            },
        })
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let Ok(end_ns) = crate::serve::phase_trace::monotonic_ns() else {
            return;
        };
        eprintln!("{PREFIX}{}", record(self.begin_ns, end_ns, &self.detail));
    }
}

fn record(begin_ns: u64, end_ns: u64, detail: &Detail) -> serde_json::Value {
    match detail {
        Detail::Decode { mode, lanes } => json!({
            "schema": SCHEMA,
            "clock_id": "CLOCK_MONOTONIC",
            "kind": "decode",
            "mode": mode,
            "begin_ns": begin_ns,
            "end_ns": end_ns,
            "width": lanes.len(),
            "lanes": lanes.iter().map(|lane| json!({
                "slot": lane.slot,
                "verify_eligibility": lane.verify_eligibility,
            })).collect::<Vec<_>>(),
        }),
        Detail::Prefill {
            slot,
            prompt_tokens,
            chunk_start,
            chunk_end,
        } => json!({
            "schema": SCHEMA,
            "clock_id": "CLOCK_MONOTONIC",
            "kind": "prefill",
            "begin_ns": begin_ns,
            "end_ns": end_ns,
            "slot": slot,
            "prompt_tokens": prompt_tokens,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_record_preserves_lane_eligibility() {
        let value = record(
            10,
            20,
            &Detail::Decode {
                mode: "joint",
                lanes: vec![
                    Lane {
                        slot: 1,
                        verify_eligibility: "eligible",
                    },
                    Lane {
                        slot: 3,
                        verify_eligibility: "blocked_transform",
                    },
                ],
            },
        );
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["clock_id"], "CLOCK_MONOTONIC");
        assert_eq!(value["kind"], "decode");
        assert_eq!(value["mode"], "joint");
        assert_eq!(value["width"], 2);
        assert_eq!(value["lanes"][1]["slot"], 3);
        assert_eq!(value["lanes"][1]["verify_eligibility"], "blocked_transform");
    }

    #[test]
    fn prefill_record_keeps_chunk_bounds() {
        let value = record(
            30,
            40,
            &Detail::Prefill {
                slot: 2,
                prompt_tokens: 725,
                chunk_start: 720,
                chunk_end: 725,
            },
        );
        assert_eq!(value["kind"], "prefill");
        assert_eq!(value["slot"], 2);
        assert_eq!(value["prompt_tokens"], 725);
        assert_eq!(value["chunk_start"], 720);
        assert_eq!(value["chunk_end"], 725);
    }
}
