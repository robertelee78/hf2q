#!/usr/bin/env bash
set -euo pipefail

# Run a release gate in its own process group. GitHub cancellation targets this
# supervisor directly (the workflow uses exec); the supervisor then terminates
# only that gate's descendants, including a foreground compiler or model
# server that would otherwise outlive the canceled job.

[[ $# -gt 0 ]] || {
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
}

gate_pid=""

cleanup_gate_group() {
  local original_rc=$?
  local deadline
  trap - EXIT INT TERM HUP

  if [[ -n "$gate_pid" ]] && kill -0 -- "-$gate_pid" 2>/dev/null; then
    kill -TERM -- "-$gate_pid" 2>/dev/null || true
    deadline=$((SECONDS + 15))
    while kill -0 -- "-$gate_pid" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    if kill -0 -- "-$gate_pid" 2>/dev/null; then
      kill -KILL -- "-$gate_pid" 2>/dev/null || true
    fi
  fi
  if [[ -n "$gate_pid" ]]; then
    wait "$gate_pid" 2>/dev/null || true
  fi
  exit "$original_rc"
}

trap cleanup_gate_group EXIT INT TERM HUP
set -m
"$@" &
gate_pid=$!
wait "$gate_pid"
