#!/usr/bin/env bash
# ADR-019 Phase 2 iter90b — 5x cold-process determinism harness
#
# Drives the production hf2q binary 5 times against the apex DWQ46
# fixture, with --max-tokens 1 --temperature 0 --top-k 1 --logits-trace,
# captures the top-3 logits, and asserts byte-identical across all 5
# runs.  Mirrors the iter90 manual 5x loop documented in
# `/opt/hf2q/.cfa-archive/iter90/determinism_env1.txt` (post-spec
# deferred to script per iter90b spec §6.2).
#
# Inputs (env):
#   HF2Q                       (path to hf2q binary; default
#                              /opt/hf2q/target/release/hf2q)
#   HF2Q_ENCODER_SESSION       (optional; pass through if set in the
#                              caller env)
#   MLX_UNRETAINED_REFS        (optional; pass through if set in the
#                              caller env)
#   DETERMINISM_FIXTURE        (default
#                              /opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf)
#   DETERMINISM_NUM_RUNS       (default 5)
#
# Output (stdout):
#   per-run "  run N: top-3: [...]" line
#   final "DETERMINISM: PASS" or "DETERMINISM: FAIL"
#   exit code 0 on PASS, 1 on FAIL

set -euo pipefail

HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"
FIXTURE="${DETERMINISM_FIXTURE:-/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf}"
N_RUNS="${DETERMINISM_NUM_RUNS:-5}"

PROMPT="Hello"

if [[ ! -x "$HF2Q" ]]; then
  echo "ERROR: hf2q binary not found / not executable: $HF2Q" >&2
  exit 2
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "ERROR: fixture not found: $FIXTURE" >&2
  echo "       set DETERMINISM_FIXTURE=<path> to override" >&2
  exit 2
fi

declare -a TOP3_LINES=()

for i in $(seq 1 "$N_RUNS"); do
  # Each run is a fresh hf2q process — cold load every time.  We set
  # `HF2Q_DUMP_LOGITS=1` so hf2q emits a top-3 line to stderr after
  # prefill (see `src/serve/mod.rs:2261-2268`).  With --temperature 0
  # + --top-k 1 the sampler is deterministic given byte-identical
  # logits.
  out=$(HF2Q_DUMP_LOGITS=1 "$HF2Q" generate \
        --model "$FIXTURE" \
        --prompt "$PROMPT" \
        --max-tokens 1 \
        --temperature 0 \
        --top-k 1 2>&1 || true)

  # Extract the "top-3:" line from the dump.  Format:
  # "  top-3: [(tok1, l1), (tok2, l2), (tok3, l3)]"
  top3=$(echo "$out" | grep -E 'top-3:' | head -1 || true)
  if [[ -z "$top3" ]]; then
    echo "  run $i: WARN no top-3 line in output (run failed?)" >&2
    # Capture last 5 stderr lines for diagnostics.
    diag=$(echo "$out" | tail -5 | tr '\n' ' ')
    top3="ERROR: $diag"
  fi
  echo "  run $i: $top3"
  TOP3_LINES+=("$top3")
done

# Compare all 5 lines.
PASS=true
first="${TOP3_LINES[0]}"
for line in "${TOP3_LINES[@]:1}"; do
  if [[ "$line" != "$first" ]]; then
    PASS=false
    break
  fi
done

if [[ "$PASS" == "true" ]]; then
  echo "DETERMINISM: PASS — $N_RUNS/$N_RUNS cold runs produced byte-identical top-3"
  exit 0
else
  echo "DETERMINISM: FAIL — $N_RUNS/$N_RUNS cold runs DID NOT produce byte-identical top-3"
  for i in "${!TOP3_LINES[@]}"; do
    echo "  run $((i+1)): ${TOP3_LINES[$i]}"
  done
  exit 1
fi
