#!/usr/bin/env bash
# scripts/iter37-parity-matrix.sh
#
# ADR-015 iter37 — minimal parity matrix for HF2Q_AUTO_BARRIER env-gate.
#
# iter37 ports llama.cpp's mem_ranges dataflow check into mlx-native and
# exposes it via `CommandEncoder::dispatch_tracked_*` + the
# `HF2Q_AUTO_BARRIER=1` env gate.  Because no production callsite in
# hf2q has migrated to the new API yet (iter38+ scope), the env gate is
# structurally a no-op: the gate-on branch is dead code from hf2q's
# perspective.  This matrix is the *durability* check: build hf2q
# against the patched mlx-native and prove that linking the new module
# in does not regress decode parity.
#
# Per fixture × per trial:
#   - Run hf2q generate --temperature 0 --max-tokens NGEN with
#     HF2Q_AUTO_BARRIER=1 and again with HF2Q_AUTO_BARRIER unset.
#   - Save stdout to a file.
#   - The two stdout files MUST be byte-identical (sha256 match).
#
# Usage:
#   N_TRIALS=3 NGEN=64 scripts/iter37-parity-matrix.sh
#
# Env:
#   N_TRIALS  (default 3)
#   NGEN      (default 64)        — keep small; this is parity, not perf
#   FIXTURES  (comma-sep cell labels; default "qwen3.6-27b-dwq46,gemma-26B-dwq")

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/.claude/worktrees/agent-ada74dd0901d1f202/target/release/hf2q}"
N_TRIALS="${N_TRIALS:-3}"
NGEN="${NGEN:-64}"
PROMPT="${PROMPT:-Hello, my name is}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter37/parity}"
FIXTURES="${FIXTURES:-qwen3.6-27b-dwq46,gemma-26B-dwq}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/parity-${DATE_TAG}.md"

declare -A FIXTURE_PATHS=(
  [qwen3.6-27b-dwq46]="/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf"
  [qwen3.6-35b-a3b-dwq46]="/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf"
  [qwen3.6-35b-a3b-apex]="/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf"
  [gemma-26B-dwq]="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
)

[[ -x "$HF2Q_BIN" ]] || { echo "FAIL: HF2Q_BIN missing: $HF2Q_BIN" >&2; exit 1; }

HF2Q_GIT_HEAD=$(git -C "$(dirname "$(dirname "$(dirname "$HF2Q_BIN")")")" rev-parse HEAD 2>/dev/null || echo unknown)
MLX_GIT_HEAD=$(git -C /opt/mlx-native/.cfa-worktrees/iter37-mem-ranges rev-parse HEAD 2>/dev/null || echo unknown)

echo "=== ADR-015 iter37 parity matrix run $DATE_TAG ==="
echo "hf2q HEAD : $HF2Q_GIT_HEAD"
echo "mlx HEAD  : $MLX_GIT_HEAD"
echo "binary    : $HF2Q_BIN"
echo "n_trials  : $N_TRIALS"
echo "n_gen     : $NGEN"
echo "fixtures  : $FIXTURES"
echo "prompt    : $PROMPT"
echo "out       : $OUT_DIR"
echo

{
  echo "# ADR-015 iter37 parity matrix"
  echo
  echo "**UTC:** $DATE_TAG  "
  echo "**hf2q HEAD:** \`$HF2Q_GIT_HEAD\`  "
  echo "**mlx-native HEAD:** \`$MLX_GIT_HEAD\`  "
  echo "**Trials per cell:** $N_TRIALS  "
  echo "**NGEN:** $NGEN  "
  echo "**Prompt:** \`$PROMPT\`  "
  echo
  echo "## Matrix"
  echo
  echo "| fixture | trial | env-off sha256 | env-on sha256 | byte-identical? |"
  echo "|---|---:|---|---|---:|"
} > "$SUMMARY"

OVERALL_PASS=1
IFS=',' read -ra WANTED <<< "$FIXTURES"
for label in "${WANTED[@]}"; do
  model="${FIXTURE_PATHS[$label]:-}"
  if [[ -z "$model" || ! -f "$model" ]]; then
    echo "[SKIP] $label — model not found"
    echo "| $label | — | MISSING | MISSING | SKIP |" >> "$SUMMARY"
    continue
  fi

  echo
  echo "[fixture] $label"
  echo "    model: $model"

  fixture_dir="$OUT_DIR/$label"
  mkdir -p "$fixture_dir"

  for trial in $(seq 1 "$N_TRIALS"); do
    off_out="$fixture_dir/trial-${trial}-env-off.txt"
    on_out="$fixture_dir/trial-${trial}-env-on.txt"

    # Run with HF2Q_AUTO_BARRIER unset.
    env -u HF2Q_AUTO_BARRIER \
      "$HF2Q_BIN" generate \
      --model "$model" \
      --prompt "$PROMPT" \
      --temperature 0 \
      --max-tokens "$NGEN" \
      > "$off_out" 2>"$fixture_dir/trial-${trial}-env-off.stderr"

    # Run with HF2Q_AUTO_BARRIER=1.
    HF2Q_AUTO_BARRIER=1 \
      "$HF2Q_BIN" generate \
      --model "$model" \
      --prompt "$PROMPT" \
      --temperature 0 \
      --max-tokens "$NGEN" \
      > "$on_out" 2>"$fixture_dir/trial-${trial}-env-on.stderr"

    # Strip the timing prologue (everything up to and including the
    # first blank line — `hf2q generate` emits 3 lines of metadata
    # ("loaded in 6.2s", "prefill: ...ms") before the decoded text).
    # Then sha the decoded-token suffix only — wall-clock noise
    # (model load time, prefill ms) is not a parity signal.
    off_sha=$(awk 'p; /^$/ && !p {p=1}' "$off_out" | shasum -a 256 | awk '{print $1}' | cut -c1-12)
    on_sha=$(awk 'p; /^$/ && !p {p=1}' "$on_out" | shasum -a 256 | awk '{print $1}' | cut -c1-12)

    if [[ "$off_sha" == "$on_sha" ]]; then
      verdict="PASS"
    else
      verdict="FAIL"
      OVERALL_PASS=0
    fi
    echo "    trial $trial: env-off=$off_sha env-on=$on_sha → $verdict"
    echo "| $label | $trial | $off_sha | $on_sha | $verdict |" >> "$SUMMARY"
  done
done

{
  echo
  echo "## Verdict"
  echo
  if (( OVERALL_PASS == 1 )); then
    echo "**OVERALL: PASS** — all trials byte-identical between HF2Q_AUTO_BARRIER=1 and unset."
    echo
    echo "Expected: with no production callsite migrated to the \`dispatch_tracked_*\` family in iter37, the env gate is a no-op from hf2q's perspective. This matrix confirms that linking the new \`mem_ranges\` module + encoder modifications did not regress decode parity."
  else
    echo "**OVERALL: FAIL** — at least one trial diverged."
    echo
    echo "Investigate: there should be NO live callsite that exercises the new env-gated branch in iter37, so any divergence is a link-time accident (e.g. clobbered counter, accidental codegen drift)."
  fi
} >> "$SUMMARY"

echo
echo "=== summary written to $SUMMARY ==="
cat "$SUMMARY"

if (( OVERALL_PASS == 0 )); then
  exit 1
fi
