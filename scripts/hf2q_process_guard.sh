#!/usr/bin/env bash

# Shared process classifier for the large-model launchers. `pgrep -x hf2q`
# alone also matches conversion, inspection, and cache-management commands;
# those processes do not own an inference model and must not trip the
# one-server-at-a-time memory guard.

hf2q_command_runs_serve() {
    local command_line=${1:-}
    [[ "$command_line" =~ (^|[[:space:]])serve([[:space:]]|$) ]]
}

hf2q_active_serve_pids() {
    local pid command_line

    command -v pgrep >/dev/null 2>&1 || return 0
    command -v ps >/dev/null 2>&1 || return 0
    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        command_line=$(ps -ww -p "$pid" -o command= 2>/dev/null || true)
        if hf2q_command_runs_serve "$command_line"; then
            printf '%s\n' "$pid"
        fi
    done < <(pgrep -x hf2q 2>/dev/null || true)
}
