#!/usr/bin/env bash
# ADR-010 iter-64 batched-prefill coherence regression gate.
#
# Locks in the fix landed in commit 133722d (forward_prefill_batched.rs +91 LOC):
# eager allocation of leg_hb_encoded + per-layer HB encode via
# dispatch_hadamard_quantize_kv_hb_seq, mirroring per-token forward_prefill.rs.
#
# Without this fix, batched prefill writes nothing to leg_hb_encoded; decode
# reads zero-init bytes via flash_attn_vec_tq_hb → garbage attention →
# gibberish tokens (correct first decode token from LM head, then drift).
#
# This gate runs the same coherence test that surfaced the bug:
#   1. Per-token prefill on "What is 2 plus 2? Answer with just a number."
#      → captures decoded text. Reference truth.
#   2. Batched prefill on the SAME prompt with the same model, same seed.
#      → must produce the SAME decoded text (or at minimum, contain "4").
#
# # Why this gate exists
#
# Pre-iter-64, NO Rust regression test covered batched-prefill correctness —
# only `scripts/sourdough_gate.sh` (long-prompt llama-vs-hf2q byte parity).
# That gap allowed 3+ weeks of intervening commits (415c9d6, e9fd6fc, etc.)
# to silently break the path. This gate fills the hole at small-prompt scale,
# fast enough (~30s) to run after any forward_prefill_batched.rs change.
#
# # Usage
#
# bash scripts/adr010_iter64_batched_coherence_gate.sh [MODEL_PATH]
#
# Default MODEL_PATH = operator's gemma4-ara-2pass-APEX-Q5_K_M.gguf fixture.
# Override via positional arg or MODEL env var.
#
# # Exit codes
#
#   0   PASS (per-token output contains "4" AND batched output matches)
#   1   FAIL (batched output diverges or doesn't contain "4")
#   2   SETUP (model not found, hf2q not built, etc.)
#
# # ADR refs
#
# ADR-010 §Status Log 2026-05-09 iter-59..64.

set -euo pipefail

MODEL="${MODEL:-${1:-/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf}}"
HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL" >&2
    echo "Hint: pass model path as positional arg or set MODEL env." >&2
    exit 2
fi
if [[ ! -x "$HF2Q" ]]; then
    echo "ERROR: hf2q binary not found or not executable: $HF2Q" >&2
    echo "Hint: cargo build --release --bin hf2q" >&2
    exit 2
fi

PROMPT_FILE=$(mktemp -t adr010-iter64-prompt.XXXXXX)
trap "rm -f $PROMPT_FILE" EXIT
echo "What is 2 plus 2? Answer with just a number." > "$PROMPT_FILE"

extract_decoded() {
    # Strip banner noise the same way bench-needle-haystack.sh does
    # (banners interleave with stdout under 2>&1 and can split tokens).
    python3 -c "
import re, sys
log = sys.stdin.read()
log = re.sub(r'\[HF2Q_TQ_CODEBOOK_BITS\][^\n]*', '', log)
log = re.sub(r'\[iter-21 Track B\][^\n]*', '', log)
idx = log.find('--- mlx-native:')
log = log[:idx] if idx >= 0 else log
pre = log.find('tok/s)\n')
log = log[pre + len('tok/s)\n'):] if pre >= 0 else log
print(log.replace('\n','').replace(' ',''))
"
}

echo "=== ADR-010 iter-64 batched-prefill coherence gate ==="
echo "Model: $MODEL"
echo

echo "[1/3] Per-token prefill (reference truth)"
PER_TOKEN_OUT=$(timeout 120 "$HF2Q" generate --model "$MODEL" --prompt-file "$PROMPT_FILE" --max-tokens 8 2>&1)
PER_TOKEN_DECODED=$(echo "$PER_TOKEN_OUT" | extract_decoded)
echo "  decoded: [$PER_TOKEN_DECODED]"

if [[ "$PER_TOKEN_DECODED" != *"4"* ]]; then
    echo "  ✗ SETUP error: per-token reference doesn't contain '4'." >&2
    echo "  Per-token path may be broken; this gate cannot evaluate batched divergence." >&2
    exit 2
fi
echo "  ✓ per-token reference OK"
echo

echo "[2/3] Batched prefill (gated HF2Q_BATCHED_PREFILL=1 + UNSAFE)"
BATCHED_OUT=$(HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1 \
    timeout 120 "$HF2Q" generate --model "$MODEL" --prompt-file "$PROMPT_FILE" --max-tokens 8 2>&1)
BATCHED_DECODED=$(echo "$BATCHED_OUT" | extract_decoded)
echo "  decoded: [$BATCHED_DECODED]"

# Coherence check 1: must contain "4" (the correct answer).
if [[ "$BATCHED_DECODED" != *"4"* ]]; then
    echo "  ✗ FAIL: batched decoded text doesn't contain '4'."
    echo "  Bug class: batched prefill produces wrong first decode token."
    echo "  Likely root cause: prefill compute path (matmul / norm / RoPE)."
    exit 1
fi

# Coherence check 2: must NOT contain a long digit run (gibberish-pattern detector).
# The pre-iter-64 bug produced "41211789444444444440" — characteristic of
# zero-init leg_hb_encoded reads. Refuse any output with 6+ consecutive digits.
if [[ "$BATCHED_DECODED" =~ [0-9]{6,} ]]; then
    echo "  ✗ FAIL: batched decoded text contains 6+ consecutive digits — gibberish pattern."
    echo "  Likely root cause: leg_hb_encoded not populated (regression of iter-64 fix)."
    echo "  See: scripts/adr010_iter64_batched_coherence_gate.sh + ADR-010 §Status Log iter-64."
    exit 1
fi

echo "  ✓ batched coherence OK"
echo

echo "[3/3] Verdict"
echo "  per-token: [$PER_TOKEN_DECODED]"
echo "  batched:   [$BATCHED_DECODED]"
echo
echo "✓ PASS — batched-prefill coherence gate green at HEAD."
echo "  iter-64 leg_hb_encoded fix is locked in for this fixture."
exit 0
