//! Process-unique OpenAI tool-call ids (`call_hf2q_<16 hex>`).
//!
//! Clients echo the id in their `tool_call_id` follow-up message, so it must
//! never repeat across concurrent requests or across calls in one
//! conversation. Ids built from wall-clock nanoseconds collided between
//! concurrent agents. Here an odd-multiplier counter is a bijection over
//! `u64` (unique for 2^64 calls per process), and a random per-process salt
//! keeps ids from a restarted server distinct from ones already in a
//! client's history.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

const MIX: u64 = 0x9e37_79b9_7f4a_7c15;

static SALT: LazyLock<u64> = LazyLock::new(rand::random::<u64>);

/// The next tool-call id for this process.
pub(crate) fn next_tool_call_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("call_hf2q_{:016x}", SALT.wrapping_add(n.wrapping_mul(MIX)))
}

#[cfg(test)]
mod tests {
    use super::next_tool_call_id;
    use std::collections::HashSet;

    #[test]
    fn concurrent_tool_call_ids_never_collide() {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| (0..5_000).map(|_| next_tool_call_id()).collect::<Vec<_>>())
            })
            .collect();
        let mut seen = HashSet::new();
        for t in threads {
            for id in t.join().unwrap() {
                assert!(
                    id.starts_with("call_hf2q_") && id.len() == "call_hf2q_".len() + 16,
                    "{id}"
                );
                assert!(seen.insert(id.clone()), "duplicate tool-call id {id}");
            }
        }
        assert_eq!(seen.len(), 40_000);
    }
}
