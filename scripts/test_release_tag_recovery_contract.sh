#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELPER="$ROOT_DIR/scripts/ensure_github_release_tag.sh"
RELEASE_WORKFLOW="$ROOT_DIR/.github/workflows/release.yml"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-release-tag-contract.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

fail() {
  echo "$*" >&2
  exit 1
}

mkdir -p "$scratch/bin"
cat >"$scratch/bin/gh" <<'FAKE_GH'
#!/usr/bin/env bash
set -euo pipefail

echo "$*" >>"$FAKE_GH_CALLS"

expected=${FAKE_GH_EXPECTED_SHA:?}
wrong=1111111111111111111111111111111111111111
state=missing
[[ ! -f "$FAKE_GH_STATE" ]] || state=$(<"$FAKE_GH_STATE")

is_post=0
for arg in "$@"; do
  [[ "$arg" != POST ]] || is_post=1
done

if [[ "$is_post" -eq 1 ]]; then
  case "$FAKE_GH_SCENARIO" in
    create-failure)
      echo '{"message":"Service unavailable","status":"503"}'
      echo 'gh: Service unavailable (HTTP 503)' >&2
      exit 1
      ;;
    create-wrong)
      printf '%s\n' created >"$FAKE_GH_STATE"
      printf '%s\n' "$wrong"
      exit 0
      ;;
    *)
      printf '%s\n' created >"$FAKE_GH_STATE"
      printf '%s\n' "$expected"
      exit 0
      ;;
  esac
fi

case "$FAKE_GH_SCENARIO" in
  existing-correct)
    printf '%s\n' "$expected"
    ;;
  existing-wrong)
    printf '%s\n' "$wrong"
    ;;
  non-404)
    echo '{"message":"Service unavailable","status":"503"}'
    echo 'gh: Service unavailable (HTTP 503)' >&2
    exit 1
    ;;
  multi-document-404)
    echo '{"message":"Service unavailable","status":"503"}'
    echo '{"message":"Not Found","status":"404"}'
    echo 'gh: conflicting API failure documents' >&2
    exit 1
    ;;
  missing|create-failure|create-wrong|reread-failure)
    if [[ "$state" == missing ]]; then
      echo '{"message":"Not Found","status":"404"}'
      echo 'gh: Not Found (HTTP 404)' >&2
      exit 1
    fi
    if [[ "$FAKE_GH_SCENARIO" == reread-failure ]]; then
      echo '{"message":"Service unavailable","status":"503"}'
      echo 'gh: Service unavailable (HTTP 503)' >&2
      exit 1
    fi
    printf '%s\n' "$expected"
    ;;
  reread-wrong)
    if [[ "$state" == missing ]]; then
      echo '{"message":"Not Found","status":"404"}'
      echo 'gh: Not Found (HTTP 404)' >&2
      exit 1
    fi
    printf '%s\n' "$wrong"
    ;;
  *)
    echo "unknown fake scenario: $FAKE_GH_SCENARIO" >&2
    exit 2
    ;;
esac
FAKE_GH
chmod 0755 "$scratch/bin/gh"

expected_sha=0123456789abcdef0123456789abcdef01234567

run_case() {
  local scenario=$1
  local expectation=$2
  local case_dir="$scratch/$scenario"
  local rc=0

  mkdir -p "$case_dir"
  : >"$case_dir/calls"
  printf '%s\n' missing >"$case_dir/state"
  PATH="$scratch/bin:$PATH" \
    FAKE_GH_SCENARIO="$scenario" \
    FAKE_GH_EXPECTED_SHA="$expected_sha" \
    FAKE_GH_CALLS="$case_dir/calls" \
    FAKE_GH_STATE="$case_dir/state" \
    "$HELPER" example/hf2q v0.1.7 "$expected_sha" \
    >"$case_dir/stdout" 2>"$case_dir/stderr" || rc=$?

  if [[ "$expectation" == pass ]]; then
    [[ "$rc" -eq 0 ]] || fail "$scenario unexpectedly failed"
  else
    [[ "$rc" -ne 0 ]] || fail "$scenario unexpectedly passed"
  fi
}

run_case missing pass
[[ "$(wc -l <"$scratch/missing/calls" | tr -d ' ')" -eq 3 ]] || \
  fail "missing-tag path did not perform GET, POST, re-read"
grep -q -- '--method POST' "$scratch/missing/calls" || \
  fail "missing-tag path did not create the tag"
grep -qF "created and verified tag v0.1.7 at $expected_sha" \
  "$scratch/missing/stdout" || fail "missing-tag path did not report verification"

run_case existing-correct pass
[[ "$(wc -l <"$scratch/existing-correct/calls" | tr -d ' ')" -eq 1 ]] || \
  fail "existing correct tag performed a mutation"
if grep -q -- '--method POST' "$scratch/existing-correct/calls"; then
  fail "existing correct tag was recreated"
fi

run_case existing-wrong fail
[[ "$(wc -l <"$scratch/existing-wrong/calls" | tr -d ' ')" -eq 1 ]] || \
  fail "wrong existing tag performed a mutation"
grep -qF 'points to 1111111111111111111111111111111111111111' \
  "$scratch/existing-wrong/stderr" || fail "wrong existing tag lacked a diagnostic"

run_case non-404 fail
[[ "$(wc -l <"$scratch/non-404/calls" | tr -d ' ')" -eq 1 ]] || \
  fail "non-404 lookup failure performed a mutation"
grep -qF 'expected an explicit 404' "$scratch/non-404/stderr" || \
  fail "non-404 lookup failure lacked a diagnostic"

run_case multi-document-404 fail
[[ "$(wc -l <"$scratch/multi-document-404/calls" | tr -d ' ')" -eq 1 ]] || \
  fail "multi-document lookup failure performed a mutation"
grep -qF 'expected an explicit 404' "$scratch/multi-document-404/stderr" || \
  fail "multi-document lookup failure lacked a diagnostic"

run_case create-failure fail
[[ "$(wc -l <"$scratch/create-failure/calls" | tr -d ' ')" -eq 2 ]] || \
  fail "create failure did not stop after POST"
grep -qF 'failed to create tag v0.1.7' "$scratch/create-failure/stderr" || \
  fail "create failure lacked a diagnostic"

run_case create-wrong fail
[[ "$(wc -l <"$scratch/create-wrong/calls" | tr -d ' ')" -eq 2 ]] || \
  fail "wrong create response continued to re-read"
grep -qF 'created tag v0.1.7 points to' "$scratch/create-wrong/stderr" || \
  fail "wrong create response lacked a diagnostic"

run_case reread-wrong fail
[[ "$(wc -l <"$scratch/reread-wrong/calls" | tr -d ' ')" -eq 3 ]] || \
  fail "re-read mismatch did not execute the complete create path"
grep -qF 're-read tag v0.1.7 points to' "$scratch/reread-wrong/stderr" || \
  fail "re-read mismatch lacked a diagnostic"

run_case reread-failure fail
[[ "$(wc -l <"$scratch/reread-failure/calls" | tr -d ' ')" -eq 3 ]] || \
  fail "re-read API failure did not execute the complete create path"
grep -qF 'failed to re-read created tag v0.1.7' \
  "$scratch/reread-failure/stderr" || \
  fail "re-read API failure lacked a diagnostic"

grep -qF 'bash scripts/ensure_github_release_tag.sh' "$RELEASE_WORKFLOW" || \
  fail "release workflow does not invoke the tested tag helper"
# shellcheck disable=SC2016
if grep -qF 'tag_sha=$(gh api' "$RELEASE_WORKFLOW"; then
  fail "release workflow still contains the fail-open tag lookup"
fi

echo "release tag recovery contract: PASS"
