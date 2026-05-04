#!/usr/bin/env bash
# ADR-019 Phase 2 iter91 Worker B — H2 production CB-count probe (AC-2)
#
# Runs the production hf2q binary on a chunk-engaged Qwen3.6 27B-DWQ46
# prefill once with HF2Q_ENCODER_SESSION=0 + HF2Q_DUMP_CB_COUNT=1 (the
# Plain-shape baseline, byte-identical to pre-iter91), once with
# HF2Q_ENCODER_SESSION=1 + HF2Q_DUMP_CB_COUNT=1 (the borrowed-session
# multi-stage chain). Parses the
#   `hf2q::cb_count: forward_gpu_impl pre=N post=M delta=D seq_len=S`
# line from each stderr, computes ratio = delta_env1 / delta_env0,
# prints PASS/FAIL.
#
# PASS criterion (spec §7 AC-2): ratio <= 0.70 (≥30% reduction).
# Both delta values must be > 0 (negative AC-2 if either is 0).
#
# AC-3 sister script: scripts/iter91-pp4096-wall.sh (wall-time at pp4096).
#
# Usage: bash /opt/hf2q/.cfa-worktrees/iter91-claude/scripts/iter91-cb-count-probe.sh
#
# Env overrides:
#   HF2Q       — path to hf2q binary
#   FIXTURE    — path to GGUF
#   PROMPT     — path to prompt file
set -euo pipefail

HF2Q="${HF2Q:-/opt/hf2q/.cfa-worktrees/iter91-claude/target/release/hf2q}"
FIXTURE="${FIXTURE:-/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf}"
PROMPT="${PROMPT:-/tmp/iter91-cb-probe-prompt.txt}"

if [[ ! -x "${HF2Q}" ]]; then
  echo "ERROR: HF2Q binary not found at ${HF2Q}" >&2
  exit 2
fi
if [[ ! -f "${FIXTURE}" ]]; then
  echo "ERROR: GGUF fixture not found at ${FIXTURE}" >&2
  exit 2
fi
if [[ ! -f "${PROMPT}" ]]; then
  echo "ERROR: prompt file not found at ${PROMPT}" >&2
  exit 2
fi

LOGDIR="${LOGDIR:-/tmp/iter91-cb-count-probe}"
mkdir -p "${LOGDIR}"

run_one() {
  local label="$1"
  local session_env="$2"
  local stamp="${LOGDIR}/${label}"
  echo "=== Running ${label} (HF2Q_ENCODER_SESSION=${session_env}) at $(date '+%H:%M:%S') ===" >&2
  HF2Q_ENCODER_SESSION="${session_env}" \
  HF2Q_DUMP_CB_COUNT=1 \
    "${HF2Q}" generate \
      --model "${FIXTURE}" \
      --prompt-file "${PROMPT}" \
      --max-tokens 4 \
      --temperature 0.0 \
      --top-k 1 \
      > "${stamp}.out" 2> "${stamp}.err" || {
        echo "  FAIL: ${label} (see ${stamp}.err)" >&2
        return 1
      }
  echo "  ${label} OK" >&2
}

run_one env0 0
run_one env1 1

# Parse `hf2q::cb_count: forward_gpu_impl pre=N post=M delta=D seq_len=S`
# Use the LAST occurrence in the log (forward_gpu_impl is called once per
# prefill but may be called multiple times if --max-tokens drives extra
# decode prefills; this run uses --max-tokens 4 so the prefill line is
# emitted once and the 3 subsequent decode tokens go through the per-token
# greedy path which has its own seq_len=1 prefill — we want the LAST
# multi-token call's delta, which is the prefill).
parse_delta() {
  local logfile="$1"
  # Filter for lines where seq_len > 1 (true prefill), take the first such
  # line — the very first multi-token forward_gpu_impl call IS the prefill.
  # Field layout (split on space and '='):
  #   $1=hf2q::cb_count:  $2=forward_gpu_impl  $3=pre  $4=N
  #   $5=post  $6=M  $7=delta  $8=D  $9=seq_len  $10=S
  awk -F'[ =]' '
    /^hf2q::cb_count: forward_gpu_impl/ {
      delta=$8; seq=$10;
      if (seq+0 > 1) { print delta; exit }
    }
  ' "$logfile"
}

DELTA_ENV0=$(parse_delta "${LOGDIR}/env0.err" || true)
DELTA_ENV1=$(parse_delta "${LOGDIR}/env1.err" || true)

if [[ -z "${DELTA_ENV0}" || -z "${DELTA_ENV1}" ]]; then
  echo "ERROR: failed to parse cb_count delta from log(s)" >&2
  echo "env0 log tail:" >&2
  tail -30 "${LOGDIR}/env0.err" >&2 || true
  echo "env1 log tail:" >&2
  tail -30 "${LOGDIR}/env1.err" >&2 || true
  exit 3
fi

echo ""
echo "fixture=${FIXTURE}"
echo "prompt=${PROMPT}"
echo "cb_count_env0=${DELTA_ENV0}"
echo "cb_count_env1=${DELTA_ENV1}"

# Compute ratio with python3 (bash float arith is ugly; awk is fine too).
RATIO=$(awk -v a="${DELTA_ENV1}" -v b="${DELTA_ENV0}" 'BEGIN{ if (b+0==0) print "NaN"; else printf "%.4f", a/b }')
REDUCTION_PCT=$(awk -v a="${DELTA_ENV1}" -v b="${DELTA_ENV0}" 'BEGIN{ if (b+0==0) print "NaN"; else printf "%.2f", (1.0 - a/b) * 100.0 }')
echo "ratio=${RATIO}"
echo "reduction_pct=${REDUCTION_PCT}"

if [[ "${RATIO}" == "NaN" ]] || [[ "${DELTA_ENV0}" -le 0 ]] || [[ "${DELTA_ENV1}" -le 0 ]]; then
  echo "FAIL: ratio undefined or non-positive deltas (env0=${DELTA_ENV0} env1=${DELTA_ENV1})"
  exit 1
fi

# Bash's floating-point comparison via awk → 0 = pass, 1 = fail.
if awk -v r="${RATIO}" 'BEGIN{ exit (r <= 0.70) ? 0 : 1 }'; then
  echo "PASS: ratio ${RATIO} <= 0.70 (≥30% reduction)"
  exit 0
else
  echo "FAIL: ratio ${RATIO} > 0.70 (less than 30% reduction)"
  echo "  → wire-up may not be activating multi-stage chain in the per-layer loop"
  echo "  → check enc.carry_into_next_stage Sessioned arm is memory_barrier() not fence_or_commit"
  exit 1
fi
