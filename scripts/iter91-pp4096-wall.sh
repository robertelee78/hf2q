#!/usr/bin/env bash
# ADR-019 Phase 2 iter91 Worker B — H3 chunk-engaged pp4096 wall-time guard (AC-3)
#
# Runs the production hf2q binary on a chunk-engaged Qwen3.6 27B-DWQ46
# pp4096 prefill, 3 trials each at HF2Q_ENCODER_SESSION=0 (Plain baseline)
# and HF2Q_ENCODER_SESSION=1 (borrowed-session multi-stage chain), takes
# the MIN of each, computes ratio = wall_env1_min / wall_env0_min, prints
# PASS/FAIL.
#
# PASS criterion (spec §7 AC-3): ratio <= 2.0 (no order-of-magnitude
# regression). The H3 hypothesis is that the borrowed-session shape DOES
# NOT reproduce the iter90b worker β "14-min Metal back-pressure hang"
# antipattern.
#
# Sister script: scripts/iter91-cb-count-probe.sh (CB-count reduction).
#
# Note on SoC contamination: any concurrent llama-cli processes contaminate
# both env=0 and env=1 walls SYMMETRICALLY, so the RATIO remains a valid
# AC-3 signal even under contention. The script checks for and reports
# llama-cli contamination but does NOT kill it (per Worker B brief).
#
# Usage: bash /opt/hf2q/.cfa-worktrees/iter91-claude/scripts/iter91-pp4096-wall.sh
#
# Env overrides: HF2Q, FIXTURE, PROMPT, TRIALS
set -euo pipefail

HF2Q="${HF2Q:-/opt/hf2q/.cfa-worktrees/iter91-claude/target/release/hf2q}"
FIXTURE="${FIXTURE:-/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf}"
PROMPT="${PROMPT:-/tmp/iter91-pp4096-prompt.txt}"
TRIALS="${TRIALS:-3}"

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

LOGDIR="${LOGDIR:-/tmp/iter91-pp4096-wall}"
mkdir -p "${LOGDIR}"

# SoC contamination check (informational, not enforced). The `|| true` on
# the grep guards against `set -o pipefail` killing the script when no
# llama processes match (grep returns 1 on no-match).
LLAMA_PROCS=$( { ps -ax | grep -E '(llama-cli|llama-completion)' | grep -v grep || true; } | wc -l | tr -d ' ')
if [[ "${LLAMA_PROCS}" -gt 0 ]]; then
  echo "WARN: ${LLAMA_PROCS} concurrent llama processes detected — wall numbers contaminated symmetrically; ratio remains valid signal" >&2
fi

run_one() {
  local label="$1"
  local session_env="$2"
  local stamp="${LOGDIR}/${label}"
  HF2Q_ENCODER_SESSION="${session_env}" \
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
  # Extract `prefill: <N> tok in <Mms> (T tok/s)` ms field.
  # The line shape is `prefill: 4096 tok in 1234ms (3320 tok/s)`.
  local ms
  ms=$(grep -E '^prefill:' "${stamp}.out" | tail -1 \
    | awk '{ for(i=1;i<=NF;i++) if($i ~ /ms$/) { sub(/ms$/, "", $i); print $i; break } }')
  if [[ -z "${ms}" ]]; then
    echo "  FAIL: ${label} could not parse prefill ms (out tail):" >&2
    tail -10 "${stamp}.out" >&2
    return 1
  fi
  echo "${ms}"
}

declare -a WALLS_ENV0
declare -a WALLS_ENV1

# Warmup once per env to amortize cold model load (page-cache primer).
echo "=== Warmup env=0 at $(date '+%H:%M:%S') ===" >&2
run_one "warmup-env0" 0 > /dev/null
echo "=== Warmup env=1 at $(date '+%H:%M:%S') ===" >&2
run_one "warmup-env1" 1 > /dev/null

# Interleaved trials so SoC thermal + page-cache state symmetrize.
for i in $(seq 1 "${TRIALS}"); do
  echo "=== Trial ${i}/${TRIALS} env=0 at $(date '+%H:%M:%S') ===" >&2
  W0=$(run_one "trial${i}-env0" 0)
  WALLS_ENV0+=("${W0}")
  echo "  wall_env0_trial${i}_ms=${W0}" >&2

  echo "=== Trial ${i}/${TRIALS} env=1 at $(date '+%H:%M:%S') ===" >&2
  W1=$(run_one "trial${i}-env1" 1)
  WALLS_ENV1+=("${W1}")
  echo "  wall_env1_trial${i}_ms=${W1}" >&2
done

# Compute MIN of each set.
MIN_ENV0=$(printf '%s\n' "${WALLS_ENV0[@]}" | sort -n | head -1)
MIN_ENV1=$(printf '%s\n' "${WALLS_ENV1[@]}" | sort -n | head -1)

echo ""
echo "fixture=${FIXTURE}"
echo "prompt=${PROMPT}"
echo "trials=${TRIALS}"
echo "soc_llama_procs=${LLAMA_PROCS}"
echo "wall_env0_trials_ms=${WALLS_ENV0[*]}"
echo "wall_env1_trials_ms=${WALLS_ENV1[*]}"
echo "wall_env0_ms=${MIN_ENV0}"
echo "wall_env1_ms=${MIN_ENV1}"

RATIO=$(awk -v a="${MIN_ENV1}" -v b="${MIN_ENV0}" 'BEGIN{ if (b+0==0) print "NaN"; else printf "%.4f", a/b }')
echo "ratio=${RATIO}"

if [[ "${RATIO}" == "NaN" ]]; then
  echo "FAIL: ratio undefined (env0=${MIN_ENV0})"
  exit 1
fi

# PASS = ratio <= 2.0
if awk -v r="${RATIO}" 'BEGIN{ exit (r <= 2.0) ? 0 : 1 }'; then
  echo "PASS: ratio ${RATIO} <= 2.0 (no >2x wall regression)"
  exit 0
else
  echo "FAIL: ratio ${RATIO} > 2.0 (potential Metal back-pressure recurrence)"
  echo "  → STOP — wire-up unsafe to ship; iter90b worker β report §Critical findings #1 applies"
  exit 1
fi
