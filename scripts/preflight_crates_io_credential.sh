#!/usr/bin/env bash
# Fail-fast crates.io credential preflight for Release runs.
#
# /api/v1/me rejects API tokens, so this probes POST /api/v1/me/tokens, which
# authenticates the caller before refusing and writes nothing:
#   400 "cannot use an API token to create a new API token"
#       -> the token authenticates (unscoped or create-capable scope)
#   403 "this token does not have the required permissions"
#       -> the token authenticates but is scoped (e.g. publish-update for
#          hf2q only), which is what a publish token should be
#   401, 403 "authentication failed", 403 "this action requires authentication"
#       -> expired, revoked, or unrecognized: fail the release now
#   anything else (including network errors, reported as 000)
#       -> not proven; warn and continue (publish still has Trusted Publishing)
# The token is sent to curl on stdin (--header @-), never on argv.
set -euo pipefail

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "::notice::CARGO_REGISTRY_TOKEN is not set; publish relies on crates.io Trusted Publishing (exchanged in the crates-publish job)"
  exit 0
fi

body=$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/hf2q-crates-preflight.XXXXXX")
trap 'rm -f "$body"' EXIT
status=$(printf 'Authorization: %s\n' "$CARGO_REGISTRY_TOKEN" |
  curl --silent --show-error --output "$body" \
    --write-out '%{http_code}' --max-time 30 \
    --request POST https://crates.io/api/v1/me/tokens \
    --header @- \
    --header 'Content-Type: application/json' \
    --header 'User-Agent: hf2q-release-preflight (github.com/robertelee78/hf2q)' \
    --data '{"api_token":{"name":"hf2q-release-preflight-probe"}}') ||
  status=000
unset CARGO_REGISTRY_TOKEN

reject() {
  echo "::error::CARGO_REGISTRY_TOKEN is expired, revoked, or invalid (HTTP $status). Rotate the secret, or delete it once crates.io Trusted Publishing is configured." >&2
  exit 1
}

case "$status" in
  400)
    if grep -Fq 'cannot use an API token to create a new API token' "$body"; then
      echo "crates.io credential preflight: CARGO_REGISTRY_TOKEN authenticates"
      exit 0
    fi
    ;;
  403)
    if grep -Fq 'does not have the required permissions' "$body"; then
      echo "crates.io credential preflight: scoped CARGO_REGISTRY_TOKEN authenticates"
      exit 0
    fi
    if grep -Fq 'authentication failed' "$body" ||
      grep -Fq 'this action requires authentication' "$body"; then
      reject
    fi
    ;;
  401)
    reject
    ;;
esac
echo "::warning::crates.io preflight probe returned HTTP $status with an unrecognized body; credential not proven, continuing"
