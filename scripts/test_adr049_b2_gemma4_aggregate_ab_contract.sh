#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "$script_dir/.." && pwd)
runner="$script_dir/bench_adr049_b2_gemma4_aggregate_ab.sh"
verifier="$script_dir/verify_adr049_b2_gemma4_aggregate_ab.py"
python_contract="$script_dir/test_adr049_b2_gemma4_aggregate_ab_contract.py"
launcher="$script_dir/serve_gemma4_opencode.sh"

for script in "$runner" "$launcher" "$0"; do
    bash -n "$script"
done
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck -x -e SC1091 "$runner" "$launcher" "$0"
else
    echo "shellcheck is required for the Gemma B.2 model-free contract" >&2
    exit 1
fi

pycache=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-gemma-b2-pycache.XXXXXX")
cleanup() { rm -R "$pycache"; }
trap cleanup EXIT
PYTHONPYCACHEPREFIX="$pycache" python3 -m py_compile "$verifier" "$python_contract"
python3 "$python_contract"

grep -Fq 'readonly PAIRS=8' "$runner"
grep -Fq 'readonly MIN_LOWER_95_SPEEDUP=1.05' "$runner"
grep -Fq 'readonly MAX_STABLE_BOUNDARY_ROWS=192' "$runner"
grep -Fq 'readonly MAX_FALLBACK_PRODUCT_REGRESSION=1.05' "$runner"
grep -Fq 'readonly PRIME_HISTORY_WORDS=1200' "$runner"
grep -Fq 'readonly MIN_PRIME_AGGREGATE_TOKENS=4097' "$runner"
grep -Fq 'readonly TOOL_TURN_FIXED_TOKENS=103' "$runner"
grep -Fq 'readonly PAYLOAD_WORD_TOKENS=2' "$runner"
grep -Fq 'readonly MAX_TARGET_ROW_DRIFT=4' "$runner"
grep -Fq "minimum_eligible_lower_95_speedup_exclusive:\$min_lower_speedup" "$runner"
grep -Fq 'stable_rectangular_eligible_widths:[128,192]' "$runner"
grep -Fq 'scalar_fallback_widths:[256]' "$runner"
grep -Fq "if [[ \"\$arm\" == on && \"\$target\" -le \"\$MAX_STABLE_BOUNDARY_ROWS\" ]]; then" "$runner"
grep -Fq "minimum_fallback_lower_95_ratio_exclusive:(1 / \$max_fallback_regression)" "$runner"
grep -Fq 'order_stratified_bootstrap_samples:10000' "$runner"
grep -Fq "\"\$launcher\"" "$runner"
grep -Fq "env -i PATH=\"\$RUNTIME_PATH\"" "$runner"
# shellcheck disable=SC2016
grep -Fq 'qwen36_start_power_guard "$HOST_CONTENTION_GATE_OWNER_PID"' "$runner"
grep -Fq 'HF2Q_GEMMA_B2_GATE_ISOLATED=1' "$runner"
grep -Fq 'host_contention_require_isolated_gate_owner' "$runner"
grep -Fq 'owner_scope:"release-gate-process-group"' "$runner"
# shellcheck disable=SC2016
grep -Fq 'owner_pgid:$host_contention_owner_pgid,continuous:true' "$runner"
# shellcheck disable=SC2016
grep -Fq 'observed_source=$(resolve_live_power_source)' "$runner"
if grep -Fq 'pmset -g batt | rg -q' "$runner"; then
    echo "Gemma B.2 runner retains the early-match AC probe" >&2
    exit 1
fi
grep -Fq "actual_overlap:\$actual_overlap" "$runner"
grep -Fq "wave_seconds=\$(awk -v start=\"\$earliest_start\" -v end=\"\$latest_finish\"" "$runner"
if grep -Fq 'wave_started=' "$runner"; then
    echo "Gemma B.2 runner includes request construction in the measured wave" >&2
    exit 1
fi
if grep -Fq "\"\$trace_rows\" == \"\$expected_work\"" "$runner"; then
    echo "Gemma B.2 runner conflates trace boundary width with usage work" >&2
    exit 1
fi
grep -Fq 'HF2Q_PREFILL_TIMING=1' "$runner"
grep -Fq "HF2Q_MODEL_VERIFICATION_RECEIPT=\"\$model_verification_receipt\"" "$runner"
grep -Fq 'interval[0] > MIN_LOWER_CI' "$verifier"
grep -Fq 'interval[0] > MIN_FALLBACK_LOWER_CI' "$verifier"
grep -Fq 'scalar_fallback_noninferiority' "$verifier"
grep -Fq 'normalized continuation requests differ' "$verifier"
grep -Fq 'canonical results differ' "$verifier"
grep -Fq 'did not reach exactly one B4 stable rectangle' "$verifier"
grep -Fq 'SSE wire did not end with exactly one [DONE]' "$verifier"
grep -Fq 'generated-call-ids-only' "$runner"
grep -Fq 'exact-envelope-single-choice-no-reasoning-logprobs-or-continuation-tools' "$runner"
grep -Fq "[.data[] | select(.loaded == true and .id == \$id)]" "$runner"
if grep -Fq "select(.loaded == true and .id == \$id)] \\" "$runner"; then
    echo "Gemma B.2 runner passes a shell continuation into jq source" >&2
    exit 1
fi

missing_env_log="$pycache/missing-env.log"
if env -i PATH=/usr/bin:/bin bash "$runner" >"$missing_env_log" 2>&1; then
    echo "runner accepted a missing immutable identity" >&2
    exit 1
fi
grep -Fq 'absolute clean source worktree is required' "$missing_env_log"

inherited_group_log="$pycache/inherited-group.log"
if env -i PATH=/usr/bin:/bin HF2Q_GEMMA_B2_GATE_ISOLATED=1 \
    bash "$runner" >"$inherited_group_log" 2>&1; then
    echo "runner accepted a forced sentinel in an inherited process group" >&2
    exit 1
fi
grep -Fq 'calibrated leaf does not own an isolated process group' \
    "$inherited_group_log"

git -C "$source_root" diff --check
echo "ADR-049 B.2 Gemma aggregation model-free shell contract passed"
