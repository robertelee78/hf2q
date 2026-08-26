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
grep -Fq 'readonly PAYLOAD_WORD_ADJUSTMENT=40' "$runner"
grep -Fq 'readonly MAX_TARGET_ROW_DRIFT=4' "$runner"
grep -Fq "minimum_lower_95_speedup_exclusive:\$min_lower_speedup" "$runner"
grep -Fq 'order_stratified_bootstrap_samples:10000' "$runner"
grep -Fq "\"\$launcher\"" "$runner"
grep -Fq "env -i PATH=\"\$RUNTIME_PATH\"" "$runner"
grep -Fq 'qwen36_start_power_guard "$$"' "$runner"
# shellcheck disable=SC2016
grep -Fq 'observed_source=$(resolve_live_power_source)' "$runner"
if grep -Fq 'pmset -g batt | rg -q' "$runner"; then
    echo "Gemma B.2 runner retains the early-match AC probe" >&2
    exit 1
fi
grep -Fq "actual_overlap:\$actual_overlap" "$runner"
grep -Fq 'HF2Q_PREFILL_TIMING=1' "$runner"
grep -Fq "HF2Q_MODEL_VERIFICATION_RECEIPT=\"\$model_verification_receipt\"" "$runner"
grep -Fq 'interval[0] > MIN_LOWER_CI' "$verifier"
grep -Fq 'request bytes differ' "$verifier"
grep -Fq 'canonical results differ' "$verifier"
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

fixture_model="$pycache/model.gguf"
fixture_binary="$pycache/hf2q"
printf 'fixture\n' >"$fixture_model"
printf '#!/usr/bin/env bash\nexit 0\n' >"$fixture_binary"
chmod +x "$fixture_binary"
invalid_log="$pycache/invalid-launcher.log"
if MODEL="$fixture_model" MMPROJ="$pycache/missing-mmproj.gguf" \
        HF2Q_BIN="$fixture_binary" PORT=65534 HF2Q_CROSS_SLOT_ADMIT=2 \
        bash "$launcher" >"$invalid_log" 2>&1; then
    echo "Gemma launcher accepted invalid aggregation mode" >&2
    exit 1
fi
grep -Fq 'HF2Q_CROSS_SLOT_ADMIT must be 0 or 1' "$invalid_log"

if MODEL="$fixture_model" MMPROJ="$pycache/missing-mmproj.gguf" \
        HF2Q_BIN="$fixture_binary" PORT=65534 HF2Q_ADMIT_COALESCE_US=100001 \
        bash "$launcher" >"$invalid_log" 2>&1; then
    echo "Gemma launcher accepted invalid coalescing bound" >&2
    exit 1
fi
grep -Fq 'HF2Q_ADMIT_COALESCE_US must be an integer from 0 through 100000' "$invalid_log"

git -C "$source_root" diff --check
echo "ADR-049 B.2 Gemma aggregation model-free shell contract passed"
