#!/bin/sh
# peer_ref_ratchet.sh — fail-closed ratchet on literal llama.cpp references.
#
# Policy (see NOTICE): llama.cpp is credited once in NOTICE and pinned in
# data/llama_cpp_pin.txt; everywhere else prose calls it "the peer". This
# check counts lines containing the literal project name in tracked files
# outside the whitelist and fails if the count ever RISES above the
# checked-in baseline. Lower data/peer_ref_baseline.txt as cleanup lands;
# never raise it without updating NOTICE policy first.
#
# The whitelist is deliberately tiny — frozen byte-contract fixtures and the
# attribution surfaces themselves. Historical ADRs are NOT whitelisted; they
# contribute a constant that the baseline already accounts for, so any new
# mention anywhere in the tree trips the ratchet.
set -eu
cd "$(dirname "$0")/.."

WHITELIST='^(NOTICE|archive/|scripts/fixtures/|data/llama_cpp_pin\.txt|data/peer_ref_baseline\.txt|scripts/peer_ref_ratchet\.sh)'

COUNT=$(git ls-files \
  | grep -v -E "$WHITELIST" \
  | tr '\n' '\0' \
  | xargs -0 grep -I -i -c -E 'llama[._-]cpp|llamacpp' 2>/dev/null \
  | awk -F: '{s+=$NF} END {print s+0}')

BASELINE=$(cat data/peer_ref_baseline.txt)

if [ "$COUNT" -gt "$BASELINE" ]; then
  echo "FAIL: literal llama.cpp references rose to $COUNT (baseline $BASELINE)." >&2
  echo "New references belong in NOTICE-covered zones only; prose says 'the peer'." >&2
  echo "Offending-file counts (top 20):" >&2
  git ls-files | grep -v -E "$WHITELIST" | tr '\n' '\0' \
    | xargs -0 grep -I -i -c -E 'llama[._-]cpp|llamacpp' 2>/dev/null \
    | awk -F: '$NF > 0' | sort -t: -k2 -rn | head -20 >&2
  exit 1
fi

echo "peer_ref_ratchet OK: $COUNT literal references (baseline $BASELINE)."
if [ "$COUNT" -lt "$BASELINE" ]; then
  echo "note: count is below baseline — lower data/peer_ref_baseline.txt to $COUNT to lock in the progress."
fi
