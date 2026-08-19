#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
supervisor="$script_dir/run_release_gate_process_group.sh"
test_dir=$(mktemp -d -t hf2q-release-process-group.XXXXXX)
wrapper_pid=""
root_pid=""
leaf_pid=""

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

printf '%s\n' "release process-group contract: pass"
