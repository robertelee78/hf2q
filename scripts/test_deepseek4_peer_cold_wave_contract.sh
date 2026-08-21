#!/usr/bin/env bash
set -euo pipefail

# Model-free contract for the matched peer cold-wave harness.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d -t hf2q-peer-wave-contract.XXXXXX)
cleanup() {
  local rc=$?
  if ((rc != 0)); then
    find "$tmp" -maxdepth 2 -type f -print >&2 || true
    for log in "$tmp"/*.stderr "$tmp"/*/*.stderr; do
      [[ -f "$log" ]] || continue
      echo "--- $log" >&2
      sed -n '1,160p' "$log" >&2
    done
  fi
  rm -rf "$tmp"
  return "$rc"
}
trap cleanup EXIT

fake_curl="$tmp/fake-curl"
apply_fake_curl() {
  # The generated helper emulates only the curl surface used by the harness.
  # shellcheck disable=SC2016
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'output=""' \
    'while (($#)); do' \
    '  case "$1" in' \
    '    --output) output=$2; shift 2 ;;' \
    '    *) shift ;;' \
    '  esac' \
    'done' \
    '[[ -n "$output" ]]' \
    'jq -n --arg path "$FAKE_EXPECTED_PATH" --argjson prompt "$FAKE_PROMPT_TOKENS" '\''{choices:[{finish_reason:"tool_calls",message:{content:null,tool_calls:[{type:"function",function:{name:"read_file",arguments:({path:$path}|tojson)}}]}}],usage:{prompt_tokens:$prompt,completion_tokens:8,total_tokens:($prompt+8)}}'\'' >"$output"' \
    'printf '"'"'0.054000\n'"'"'' >"$fake_curl"
  chmod +x "$fake_curl"
}
apply_fake_curl

positive="$tmp/positive"
FAKE_EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
FAKE_PROMPT_TOKENS=6695 EXPECTED_PROMPT_TOKENS=6695 \
CURL_BIN="$fake_curl" AGENTS=4 \
OUT_DIR="$positive" WAVE_ID=contract-positive \
  "$ROOT_DIR/scripts/test_deepseek4_peer_cold_wave.sh" \
    >"$tmp/positive.stdout" 2>"$tmp/positive.stderr"
jq -e '
  .status == "pass"
  and .runtime == "llama.cpp"
  and .concurrent_agents == 4
  and .fixture_id == "full-context-agentic-v2"
  and (.prompt_contract_sha256 | test("^[0-9a-f]{64}$"))
  and .prompt_tokens == 6695
  and .maximum_cold_semantic_response_ms == 54
  and (.agents | length) == 4
  and all(.agents[];
    .cached_tokens == 0 and .tool_call.arguments_match == true)
' "$positive/summary.json" >/dev/null
shasum -a 256 -c "$positive/summary.json.sha256" >/dev/null

negative="$tmp/negative"
if FAKE_EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
  FAKE_PROMPT_TOKENS=6685 EXPECTED_PROMPT_TOKENS=6695 \
  CURL_BIN="$fake_curl" AGENTS=2 \
  OUT_DIR="$negative" WAVE_ID=contract-negative \
    "$ROOT_DIR/scripts/test_deepseek4_peer_cold_wave.sh" \
      >"$tmp/negative.stdout" 2>"$tmp/negative.stderr"; then
  echo "peer contract accepted a drifted prompt-token count" >&2
  exit 1
fi
grep -q 'rendered 6685 prompt tokens; expected 6695' "$tmp/negative.stderr"

echo "DeepSeek matched-peer cold-wave contract passed"
