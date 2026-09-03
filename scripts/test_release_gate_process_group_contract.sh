#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
supervisor="$script_dir/run_release_gate_process_group.sh"
release_gate="$script_dir/run_agentic_cache_release_gate.sh"
decode_gate="$script_dir/run_deepseek4_decode_cohort_gate.sh"
thermal_guard="$script_dir/macos_thermal_guard.sh"
model_workflow="$script_dir/../.github/workflows/cache-lifecycle.yml"
release_workflow="$script_dir/../.github/workflows/release.yml"
test_dir=$(mktemp -d -t hf2q-release-process-group.XXXXXX)
wrapper_pid=""
root_pid=""
leaf_pid=""
root_pgid=""
leaf_pgid=""

cleanup() {
  for pid in "$wrapper_pid" "$root_pid" "$leaf_pid"; do
    [[ -z "$pid" ]] || kill -KILL "$pid" 2>/dev/null || true
  done
  rm -rf "$test_dir"
}
trap cleanup EXIT

bash -n "$supervisor"
"$supervisor" bash -c 'exit 0'
if "$supervisor" bash -c 'exit 23'; then
  echo "process-group supervisor swallowed a child failure" >&2
  exit 1
fi

tree_script="$test_dir/tree.sh"
cat >"$tree_script" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
root_file=$1
leaf_file=$2
sleep 300 &
leaf_pid=$!
printf '%s\n' "$$" >"$root_file"
printf '%s\n' "$leaf_pid" >"$leaf_file"
wait "$leaf_pid"
SCRIPT
chmod +x "$tree_script"

"$supervisor" "$tree_script" "$test_dir/root.pid" "$test_dir/leaf.pid" &
wrapper_pid=$!
deadline=$((SECONDS + 10))
while [[ ! -s "$test_dir/root.pid" || ! -s "$test_dir/leaf.pid" ]]; do
  ((SECONDS < deadline)) || {
    echo "process-group fixture did not start" >&2
    exit 1
  }
  sleep 0.1
done
root_pid=$(cat "$test_dir/root.pid")
leaf_pid=$(cat "$test_dir/leaf.pid")
root_pgid=$(/bin/ps -p "$root_pid" -o pgid= | tr -d '[:space:]')
leaf_pgid=$(/bin/ps -p "$leaf_pid" -o pgid= | tr -d '[:space:]')
test "$root_pgid" = "$root_pid"
test "$leaf_pgid" = "$root_pgid"
kill -TERM "$wrapper_pid"
if wait "$wrapper_pid"; then
  echo "canceled process-group supervisor returned success" >&2
  exit 1
fi
wrapper_pid=""

for pid in "$root_pid" "$leaf_pid"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "canceled release descendant survived: $pid" >&2
    exit 1
  fi
done
root_pid=""
leaf_pid=""

# All calibrated DeepSeek producers must carry the same process-group evidence
# through settle, measurement, and offline receipt verification.
grep -F 'cooperative_settle_contention_log=' "$release_gate" >/dev/null
# Literal source contracts; the variable names must not expand here.
# shellcheck disable=SC2016
grep -F 'contention_settle_log="$thermal_dir/settle-contention.log"' \
  "$release_gate" >/dev/null
grep -F 'host_contention_validate_thermal_alignment' "$release_gate" >/dev/null
# shellcheck disable=SC2016
grep -F 'contention_settle_log="$out_dir/settle-contention.log"' \
  "$decode_gate" >/dev/null
grep -F 'host_contention_validate_thermal_alignment' "$decode_gate" >/dev/null
grep -F 'loaded_nominal_settle_seconds=30' "$decode_gate" >/dev/null
grep -F 'loaded_nominal_timeout_seconds=240' "$decode_gate" >/dev/null
grep -F 'loaded nominal cooldown did not remain calibrated' \
  "$decode_gate" >/dev/null
grep -F 'thermal_validate_settle_log "$setup_thermal_log"' \
  "$decode_gate" >/dev/null

# The decode monitor deliberately disables errexit while it observes the
# producer so it can capture and clean up a failed child. Every command that
# can authorize the exact measurement window must therefore propagate its own
# failure explicitly. Normalize shell continuations before checking the
# source contract so a failed probe cannot be masked by a later successful
# command.
decode_monitor=$(sed -e ':join' -e '/\\$/N;s/\\\n/ /;tjoin' "$decode_gate" |
  awk '/^monitor_decode_run\(\)/ { in_monitor=1 }
    in_monitor { print }
    in_monitor && /^}/ { exit }')
[[ -n "$decode_monitor" ]] || {
  echo "decode monitor source contract could not find monitor_decode_run" >&2
  exit 1
}
while IFS= read -r line; do
  case "$line" in
    *phase_marker_matches*|*thermal_sample*|*host_contention_sample*|\
      *host_contention_require_quiet*|*memory_sample*|\
      *capture_buffered_measurement_sample*|*flush_measurement_buffers*)
      [[ "$line" == *'|| return 1'* ]] || {
        echo "decode monitor has a fail-open measurement check: $line" >&2
        exit 1
      }
      ;;
  esac
done <<<"$decode_monitor"
grep -F 'scripts/run_release_gate_process_group.sh env' "$model_workflow" >/dev/null
grep -F 'scripts/run_agentic_cache_release_gate.sh' "$model_workflow" >/dev/null
# The release gate binds effective compute policy. Apple permits High Power
# Mode on battery, so a cable must not stand in for the actual live mode.
grep -F 'qwen36_validate_release_power_policy' "$release_gate" >/dev/null
grep -F 'QWEN36_EXPECTED_POWER_MODE_CODE' "$release_gate" >/dev/null
grep -F 'cable_required: false' "$release_gate" >/dev/null
grep -F '.power_policy.cable_required == false' \
  "$script_dir/verify_cache_lifecycle_candidate.sh" >/dev/null
if grep -Fq 'requires continuous AC power' "$release_gate"; then
  echo "release gate still substitutes an AC cable for effective power mode" >&2
  exit 1
fi
# One reference artifact for every enabled text-generation architecture must
# run the r2c compatibility gate from the sealed binary before retirement.
for family in deepseek gemma qwen qwen38; do
  test "$(grep -Ec "^[[:space:]]*run_r2c_structured_outputs $family$" "$release_gate")" = 1
done
for contract in \
  'verify_model deepseek "$DEEPSEEK_MODEL" "$DEEPSEEK_MODEL_SHA256" deepseek4' \
  'verify_model gemma "$GEMMA_MODEL" "$GEMMA_MODEL_SHA256" gemma4' \
  'verify_model qwen "$QWEN_MODEL" "$QWEN_MODEL_SHA256" qwen35moe' \
  'verify_model qwen38 "$QWEN38_MODEL" "$QWEN38_MODEL_SHA256" qwen35'; do
  grep -F "$contract" "$release_gate" >/dev/null
done
for architecture in deepseek4 gemma4 qwen35moe qwen35; do
  grep -F "architecture == \"$architecture\"" \
    "$script_dir/verify_cache_lifecycle_candidate.sh" >/dev/null
done
grep -F 'verify_r2c_structured_output_receipt.sh' "$release_gate" >/dev/null
grep -F 'extract_openai_sse_data.sh' "$script_dir/test_r2c_structured_outputs.sh" >/dev/null
grep -F 'extract_openai_sse_data.sh' "$script_dir/verify_r2c_structured_output_receipt.sh" >/dev/null
# shellcheck disable=SC2016
grep -F 'STANDALONE_CANDIDATE_RUN_ID="$EXPECTED_STANDALONE_CANDIDATE_RUN_ID"' \
  "$model_workflow" >/dev/null
grep -F 'workflow_call:' "$model_workflow" >/dev/null
grep -F 'uses: ./.github/workflows/cache-lifecycle.yml' "$release_workflow" >/dev/null
grep -F 'needs: [standalone-candidate, cache-lifecycle]' "$release_workflow" >/dev/null
grep -F 'scripts/verify_cache_lifecycle_candidate.sh' "$release_workflow" >/dev/null
grep -F 'scripts/package_cache_lifecycle_evidence.sh' "$release_workflow" >/dev/null
for suffix in qualification.tar.gz qualification.tar.gz.sha256 \
  release-proof.json release-proof.json.sha256; do
  test "$(grep -Fc "hf2q-\${EXPECTED_VERSION}-$suffix" "$release_workflow")" -ge 1
done
# shellcheck disable=SC2016
if grep -Fq '$d.cooperative_prefill.schema_version == 2' "$release_workflow"; then
  echo "publication still owns model-qualification process receipts" >&2
  exit 1
fi
if grep -Eq 'qwen|deepseek|gemma|agentic_cache|model_recipe|hf_download' \
  "$release_workflow"; then
  echo "publication still executes model-family qualification" >&2
  exit 1
fi
grep -F 'name ~ /^hf2q(-|$)/ && pgid[i] != owner_pgid' \
  "$thermal_guard" >/dev/null
if awk '
  /^host_contention_process_snapshot\(\)/ { in_guard=1 }
  /^thermal_validate_state\(\)/ { in_guard=0 }
  in_guard { print }
' "$thermal_guard" | grep -Eq '(^|[[:space:]])kill([[:space:]]|$)'; then
  echo "host contention guard attempts to signal a process" >&2
  exit 1
fi

printf '%s\n' "release process-group contract: pass"
