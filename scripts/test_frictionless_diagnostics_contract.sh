#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-frictionless-diagnostics.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

set +e
bash "$repo_root/scripts/test_frictionless_binary.sh" \
  /usr/bin/false "$scratch/smoke" \
  >"$scratch/stdout" 2>"$scratch/stderr"
exit_code=$?
set -e

test "$exit_code" -eq 1
grep -Fq 'frictionless smoke failed: phase=local-inventory' "$scratch/stderr"
grep -Eq 'line=[0-9]+ status=1 command=' "$scratch/stderr"
grep -Fq 'frictionless smoke diagnostics retained at ' "$scratch/stderr"

retained=($scratch/smoke.*)
test "${#retained[@]}" -eq 1
test -f "${retained[0]}/state/config.toml"

echo 'frictionless smoke diagnostics contract passed'
