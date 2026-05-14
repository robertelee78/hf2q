#!/usr/bin/env bash
# ADR-030 iter-68 — benchmark HF2Q_SPEC_DFLASH=1 vs baseline.
#
# Per memory rule feedback_thermal_cooldown_required_for_accurate_bench:
# 60-90s cool-downs between runs so M-series thermal throttle doesn't
# contaminate ratios.  Per feedback_one_instance_at_a_time: serial only.
#
# Output: a TSV table on stdout, raw stderr captured per run for audit.

set -uo pipefail

REPO_ROOT="${HF2Q_REPO_ROOT:-/opt/hf2q}"
HF2Q_BIN="$REPO_ROOT/target/release/hf2q"
MODEL="${HF2Q_BENCH_MODEL:-$REPO_ROOT/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf}"
PROMPT='Q: What is 2+2?\nA:'
OUTDIR="${HF2Q_BENCH_OUTDIR:-$REPO_ROOT/docs/research/adr030_iter68_bench}"
TRIALS=${HF2Q_BENCH_TRIALS:-3}
COOLDOWN=${HF2Q_BENCH_COOLDOWN:-60}

mkdir -p "$OUTDIR"

if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "BUILD: $HF2Q_BIN missing — building release" >&2
  (cd "$REPO_ROOT" && cargo build --release --bin hf2q) >&2
fi

# Extract t/s from the trailing "Generation: X t/s" stat line.
extract_gen_tok_s() {
  local file="$1"
  # mlx-native line: "--- mlx-native: [ Prompt: 257 t/s | Generation: 102.4 t/s ]  (16 gen tokens in 0.16s) ---"
  local v
  v=$(grep -oE 'Generation: [0-9]+\.[0-9]+ t/s' "$file" | tail -1 | awk '{print $2}')
  if [[ -z "$v" ]]; then
    # spec-decode path: "[HF2Q_SPEC_DFLASH] 16 new tokens in 3.77s (4.2 tok/s)"
    v=$(grep -oE '[(][0-9]+\.[0-9]+ tok/s[)]' "$file" | tail -1 | tr -d '()' | awk '{print $1}')
  fi
  echo "${v:-NA}"
}

run_one() {
  local arm="$1"  # "baseline" or "spec"
  local n="$2"
  local trial="$3"
  local logf="$OUTDIR/${arm}_N${n}_trial${trial}.log"

  printf "  %s N=%d trial=%d ..." "$arm" "$n" "$trial" >&2

  local env_prefix=""
  if [[ "$arm" == "spec" ]]; then
    env_prefix="HF2Q_SPEC_DFLASH=1"
  fi

  # shellcheck disable=SC2086
  env $env_prefix RUST_LOG=warn "$HF2Q_BIN" generate \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$n" \
    --temperature 0 \
    --ignore-eos \
    > "$logf" 2>&1

  local toks
  toks=$(extract_gen_tok_s "$logf")
  printf " %s tok/s\n" "$toks" >&2
  echo "$toks"
}

printf "arm\tN\ttrial\tgen_tok_per_s\n"

for n in 8 16 32; do
  for trial in $(seq 1 "$TRIALS"); do
    # baseline first
    res=$(run_one baseline "$n" "$trial")
    printf "baseline\t%d\t%d\t%s\n" "$n" "$trial" "$res"
    sleep "$COOLDOWN"

    # spec second
    res=$(run_one spec "$n" "$trial")
    printf "spec\t%d\t%d\t%s\n" "$n" "$trial" "$res"
    sleep "$COOLDOWN"
  done
done
