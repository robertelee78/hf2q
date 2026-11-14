#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HARNESS="$ROOT_DIR/scripts/test_gemma4_long_short_overlap.sh"
RELEASE_GATE="$ROOT_DIR/scripts/run_agentic_cache_release_gate.sh"
MODEL_WORKFLOW="$ROOT_DIR/.github/workflows/cache-lifecycle.yml"
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"
PARITY_VERIFIER="$ROOT_DIR/scripts/verify_gemma4_parity_receipt.sh"

for command in awk bash grep jq shasum; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
bash -n "$HARNESS" "$RELEASE_GATE" "$PARITY_VERIFIER"
bash -n "$ROOT_DIR/scripts/verify_gemma4_wave_thermal_receipt.sh"
grep -qF 'scripts/run_agentic_cache_release_gate.sh' "$MODEL_WORKFLOW" || {
  echo "model qualification workflow does not invoke the Gemma gate" >&2
  exit 1
}
if grep -qF '.receipt_sha256.gemma.' "$RELEASE_WORKFLOW"; then
  echo "publication still owns Gemma model-qualification receipts" >&2
  exit 1
fi

invalid_stderr=$(mktemp)
exit_probe_script=$(mktemp)
parity_contract_dir=$(mktemp -d)
cleanup_contract() {
  rm -f "$invalid_stderr" "$exit_probe_script"
  rm -rf "$parity_contract_dir"
}
trap cleanup_contract EXIT
if SERVER_PID=1 SERVER_LOG=/dev/null BINARY_PATH=/bin/true \
  BINARY_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  MODEL_PATH=/dev/null \
  MODEL_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  OUT_DIR=/dev/null CURL_MAX_TIME_SECONDS=0 \
  bash "$HARNESS" 2>"$invalid_stderr"; then
  echo "Gemma overlap accepted a non-positive curl timeout" >&2
  exit 1
fi
grep -qF 'CURL_MAX_TIME_SECONDS must be a positive integer' "$invalid_stderr"

if SERVER_PID=1 SERVER_LOG=/dev/null BINARY_PATH=/bin/true \
  BINARY_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  MODEL_PATH=/dev/null \
  MODEL_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  OUT_DIR=/dev/null CURL_MAX_TIME_SECONDS=1 CANCELLATION_WAIT_SECONDS=0 \
  bash "$HARNESS" 2>"$invalid_stderr"; then
  echo "Gemma overlap accepted a non-positive cancellation wait" >&2
  exit 1
fi
grep -qF 'CANCELLATION_WAIT_SECONDS must be a positive integer' "$invalid_stderr"

# The literal shell variable is the contract: every long curl must read the
# validated runtime value instead of embedding a fixed timeout.
# shellcheck disable=SC2016
[[ "$(grep -cF -- '--max-time "$CURL_MAX_TIME_SECONDS"' "$HARNESS")" == 3 ]] || {
  echo "every long-running Gemma curl must use the validated timeout" >&2
  exit 1
}
if grep -qF -- '--max-time 900' "$HARNESS"; then
  echo "Gemma overlap still contains a non-overridable 900-second timeout" >&2
  exit 1
fi

parity_files=(
  n4.log
  n8.log
  n8-seed-budget.log
  n8-tiny-hybrid.log
  n8-tiny-full-tq.log
  boundary-tail.log
  long-resume.log
)
for file in "${parity_files[@]}"; do
  printf 'fixture:%s\n' "$file" >"$parity_contract_dir/$file"
done
jq -n \
  --arg n4 "$(shasum -a 256 "$parity_contract_dir/n4.log" | awk '{print $1}')" \
  --arg n8 "$(shasum -a 256 "$parity_contract_dir/n8.log" | awk '{print $1}')" \
  --arg seed "$(shasum -a 256 "$parity_contract_dir/n8-seed-budget.log" | awk '{print $1}')" \
  --arg hybrid "$(shasum -a 256 "$parity_contract_dir/n8-tiny-hybrid.log" | awk '{print $1}')" \
  --arg full_tq "$(shasum -a 256 "$parity_contract_dir/n8-tiny-full-tq.log" | awk '{print $1}')" \
  --arg boundary "$(shasum -a 256 "$parity_contract_dir/boundary-tail.log" | awk '{print $1}')" \
  --arg resume "$(shasum -a 256 "$parity_contract_dir/long-resume.log" | awk '{print $1}')" \
  '{status:"pass",n4_log_sha256:$n4,n8_log_sha256:$n8,n8_seed_budget_log_sha256:$seed,n8_tiny_hybrid_log_sha256:$hybrid,n8_tiny_full_tq_log_sha256:$full_tq,boundary_tail_log_sha256:$boundary,long_resume_log_sha256:$resume}' \
  >"$parity_contract_dir/summary.json"
bash "$PARITY_VERIFIER" "$parity_contract_dir/summary.json" "$parity_contract_dir" \
  >/dev/null
printf 'unexpected\n' >"$parity_contract_dir/extra.log"
if bash "$PARITY_VERIFIER" "$parity_contract_dir/summary.json" "$parity_contract_dir" \
  >/dev/null 2>&1; then
  echo "Gemma parity verifier accepted an extra unbound log" >&2
  exit 1
fi
rm -f "$parity_contract_dir/extra.log"
printf 'tamper\n' >>"$parity_contract_dir/n8.log"
if bash "$PARITY_VERIFIER" "$parity_contract_dir/summary.json" "$parity_contract_dir" \
  >/dev/null 2>&1; then
  echo "Gemma parity verifier accepted a mutated bound log" >&2
  exit 1
fi

write_exit_probe() {
  local cleanup_status=$1
  local command_status=$2
  {
    printf '#!/usr/bin/env bash\nset -u\ncleanup() { return %s; }\n' "$cleanup_status"
    awk '/^on_exit\(\)/ { copy=1 } copy { print } copy && /^}/ { exit }' \
      "$RELEASE_GATE"
    printf 'trap on_exit EXIT\nexit %s\n' "$command_status"
  } >"$exit_probe_script"
}

write_exit_probe 0 7
if bash "$exit_probe_script"; then
  echo "release cleanup trap swallowed the originating failure" >&2
  exit 1
else
  probe_status=$?
fi
[[ "$probe_status" == 7 ]] || {
  echo "release cleanup trap changed originating status 7 to $probe_status" >&2
  exit 1
}

write_exit_probe 1 0
if bash "$exit_probe_script"; then
  echo "release cleanup failure did not fail a successful command" >&2
  exit 1
else
  probe_status=$?
fi
[[ "$probe_status" == 1 ]] || {
  echo "release cleanup failure returned unexpected status $probe_status" >&2
  exit 1
}

echo "Gemma release harness contract passed"
