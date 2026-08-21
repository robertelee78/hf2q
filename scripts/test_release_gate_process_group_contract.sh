#!/usr/bin/env bash
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
grep -F 'scripts/run_release_gate_process_group.sh env' "$model_workflow" >/dev/null
grep -F 'scripts/run_agentic_cache_release_gate.sh' "$model_workflow" >/dev/null
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
