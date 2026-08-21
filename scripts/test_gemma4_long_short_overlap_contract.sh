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

awk '
  /^run_gemma_release_gates\(\)/ { in_gemma=1 }
  in_gemma && /CURL_MAX_TIME_SECONDS=1800 CANCELLATION_WAIT_SECONDS=180/ { armed=1; next }
  armed && /scripts\/test_gemma4_long_short_overlap.sh/ { found=1; exit }
  armed && substr($0, length($0), 1) != "\\" { exit 1 }
  END { exit(found ? 0 : 1) }
' "$RELEASE_GATE"

# The two calibrated four-slot waves must run before the destructive 175K
# overlap and 120K lifecycle soak. Each wave is accepted only through the
# full-wave thermal wrapper, never the unmonitored generic helper.
awk '
  /^run_gemma_release_gates\(\)/ { in_gemma=1; next }
  in_gemma && /run_gemma_thermally_guarded_wave 1 4/ { wave1=NR }
  in_gemma && /run_gemma_thermally_guarded_wave 2 4/ { wave2=NR }
  in_gemma && /scripts\/test_gemma4_long_short_overlap\.sh/ { overlap=NR }
  in_gemma && /run_lifecycle gemma/ { lifecycle=NR }
  in_gemma && /^}/ { exit }
  END {
    exit(wave1 > 0 && wave1 < wave2 && wave2 < overlap && overlap < lifecycle ? 0 : 1)
  }
' "$RELEASE_GATE" || {
  echo "Gemma calibrated waves must precede destructive overlap/lifecycle work" >&2
  exit 1
}
# The following checks intentionally match literal shell variables.
# shellcheck disable=SC2016
grep -qF 'thermal_wait_for_nominal "$thermal_settle_log" "$phase_name-settle"' \
  "$RELEASE_GATE" || {
  echo "Gemma calibrated waves lack the nominal-settle gate" >&2
  exit 1
}
# shellcheck disable=SC2016
grep -qF 'thermal_monitor_nominal_while_pid "$thermal_measurement_log"' \
  "$RELEASE_GATE" || {
  echo "Gemma calibrated waves lack continuous full-wave thermal monitoring" >&2
  exit 1
}
if awk '
  /^run_gemma_thermally_guarded_wave\(\)/ { in_wave=1 }
  in_wave && /thermal_monitor_nominal .*&/ { found=1 }
  in_wave && /^}/ { exit }
  END { exit(found ? 0 : 1) }
' "$RELEASE_GATE"; then
  echo "Gemma calibrated waves still use a racy background thermal monitor" >&2
  exit 1
fi
grep -qF 'measurement_scope:"full-agent-wave"' "$RELEASE_GATE" || {
  echo "Gemma thermal receipt does not bind the complete agent wave" >&2
  exit 1
}
# shellcheck disable=SC2016
grep -qF 'bash scripts/verify_gemma4_wave_thermal_receipt.sh "$wave"' \
  "$RELEASE_GATE" || {
  echo "Gemma release producer does not verify its thermal receipt" >&2
  exit 1
}
awk '
  /^run_gemma_wave\(\)/ { in_wave=1 }
  in_wave && /if \[\[ "\$agents" == 8 \]\]/ { in_eight=1 }
  in_eight && /MAX_COLD_TTFT_MS=40000 MAX_COLD_RESPONSE_MS=60000/ {
    limits=1
  }
  limits && /MAX_TOOL_RESULT_RESPONSE_MS=30000/ { tool_limit=1 }
  limits && /scripts\/test_full_context_agent_slots.sh/ { eight_harness=1 }
  eight_harness && /^[[:space:]]*else[[:space:]]*$/ { in_default=1 }
  in_default && /BASE_URL="\$current_url" FAMILY=gemma4 AGENTS="\$agents"/ {
    default_env=1
  }
  default_env && /scripts\/test_full_context_agent_slots.sh/ { found=1; exit }
  END { exit(found && limits && tool_limit && eight_harness ? 0 : 1) }
' "$RELEASE_GATE"

awk '
  /^run_gemma_release_gates\(\)/ { in_gemma=1; next }
  in_gemma && /scripts\/test_gemma4_long_short_overlap\.sh/ { overlap=NR }
  in_gemma && /run_lifecycle gemma/ { lifecycle=NR }
  in_gemma && /finish_server_phase/ && lifecycle > 0 && process_a_finish == 0 {
    process_a_finish=NR
  }
  in_gemma && /start_server gemma process-b/ { process_b=NR }
  in_gemma && /run_gemma_thermally_guarded_wave eight-slots 8/ { guarded=NR }
  in_gemma && /^}/ { exit }
  END {
    ordered = overlap > 0 && overlap < lifecycle
    ordered = ordered && lifecycle < process_a_finish
    ordered = ordered && process_a_finish < process_b && process_b < guarded
    exit(ordered ? 0 : 1)
  }
' "$RELEASE_GATE" || {
  echo "Gemma eight-slot wave must be independently guarded after the process-a soak" >&2
  exit 1
}
if awk '
  /^run_gemma_release_gates\(\)/ { in_gemma=1; next }
  in_gemma && /run_gemma_wave eight-slots 8/ { found=1 }
  in_gemma && /^}/ { exit }
  END { exit(found ? 0 : 1) }
' "$RELEASE_GATE"; then
  echo "Gemma eight-slot wave still bypasses thermal supervision" >&2
  exit 1
fi
grep -qF 'gemma_wave8_thermal_sha' "$RELEASE_GATE" || {
  echo "Gemma manifest does not bind the eight-slot thermal summary" >&2
  exit 1
}
grep -qF 'jq -e -f scripts/gemma4_eight_slot_receipt.jq' \
  "$RELEASE_GATE" || {
  echo "Gemma release producer does not validate the N=8 summary before sealing" >&2
  exit 1
}
awk '
  /^on_exit\(\)/ { in_exit=1 }
  in_exit && /local original_rc=\$\?/ { captures=1 }
  captures && /trap - EXIT/ { clears=1 }
  clears && /if ! cleanup && \(\( original_rc == 0 \)\)/ { promotes_cleanup=1 }
  promotes_cleanup && /exit "\$original_rc"/ { exits=1 }
  exits && /^}/ { complete=1; exit }
  END { exit(complete ? 0 : 1) }
' "$RELEASE_GATE"

# Metal scheduling and command-buffer ordering are profile-sensitive. The
# release-authority parity suite must exercise the optimized production profile
# rather than treating a debug-only timing outcome as production evidence.
[[ "$(grep -cF 'cargo test --release --locked' "$RELEASE_GATE")" -ge 4 ]] || {
  echo "Gemma release parity checks must run in the release profile" >&2
  exit 1
}
grep -qF "profile:\"release\"" "$RELEASE_GATE" || {
  echo "Gemma parity receipt does not bind the release profile" >&2
  exit 1
}
grep -qF 'gemma_fresh_and_reused_4096_8193_bounded_outputs_match' "$RELEASE_GATE" || {
  echo "Gemma release gate does not run the bounded fresh-versus-reused parity test" >&2
  exit 1
}
[[ "$(grep -cF 'HF2Q_CROSS_SLOT_ADMIT=1 HF2Q_ADMIT_COALESCE_US=25000' "$RELEASE_GATE")" == 2 ]] || {
  echo "Gemma N=8 worker parity must use the canonical 25ms admission-coalescing window" >&2
  exit 1
}
[[ "$(grep -cF 'HF2Q_GEMMA_N8_PARITY_ROUNDS=25' "$RELEASE_GATE")" == 2 ]] || {
  echo "Gemma release gate must repeat both N=8 worker budgets 25 times" >&2
  exit 1
}
[[ "$(grep -cF 'gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant' "$RELEASE_GATE")" == 2 ]] || {
  echo "Gemma release gate must run the direct tiny-prefill discriminator in both cache regimes" >&2
  exit 1
}
for regime in \
  'HF2Q_HYBRID_KV=1 HF2Q_USE_DENSE=0 HF2Q_TQ_CODEBOOK_BITS=8' \
  'HF2Q_HYBRID_KV=0 HF2Q_USE_DENSE=0 HF2Q_TQ_CODEBOOK_BITS=8' \
  'HF2Q_GEMMA_N8_EXPECTED_KV_REGIME=hybrid' \
  'HF2Q_GEMMA_N8_EXPECTED_KV_REGIME=full-tq'; do
  grep -qF "$regime" "$RELEASE_GATE" || {
    echo "Gemma direct discriminator does not pin its requested KV regime: $regime" >&2
    exit 1
  }
done
[[ "$(grep -cF 'verify_gemma4_parity_receipt.sh' "$RELEASE_GATE")" == 2 ]] || {
  echo "Gemma producer must verify leaf parity logs after creation and before manifest seal" >&2
  exit 1
}
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
# The release script must retain the literal output-root expression.
# shellcheck disable=SC2016
[[ "$(grep -cF -- '--test-threads=1 --nocapture >"$OUT_ROOT/gemma/parity/' \
  "$RELEASE_GATE")" == 4 ]] || {
  echo "Gemma N=8 release logs must retain their non-vacuity and KV-regime banners" >&2
  exit 1
}
grep -qF 'HF2Q_GEMMA_N8_PARITY_MAX_TOKENS=24' "$RELEASE_GATE" || {
  echo "Gemma release gate does not bind the normal N=8 generation budget" >&2
  exit 1
}
grep -qF 'HF2Q_GEMMA_N8_PARITY_MAX_TOKENS=1' "$RELEASE_GATE" || {
  echo "Gemma release gate does not bind the one-token N=8 seed budget" >&2
  exit 1
}
grep -qF 'fresh_and_reused_4096_8193_bounded_output_parity:true' "$RELEASE_GATE" || {
  echo "Gemma parity receipt does not bind bounded fresh-versus-reused output" >&2
  exit 1
}
if grep -qF 'eager_4096_and_resumed_8193_exact_output_parity' \
  "$RELEASE_GATE"; then
  echo "stale monolithic-versus-bounded Gemma parity schema remains accepted" >&2
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
