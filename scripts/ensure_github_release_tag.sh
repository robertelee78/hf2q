#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <owner/repository> <tag> <expected-commit-sha>" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage

repository=$1
tag=$2
expected_sha=$3

[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || usage
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || usage
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || usage

scratch=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/hf2q-release-tag.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

lookup_stdout="$scratch/lookup.stdout"
lookup_stderr="$scratch/lookup.stderr"

fail_with_response() {
  local message=$1
  local stdout_file=$2
  local stderr_file=$3
  echo "$message" >&2
  [[ ! -s "$stdout_file" ]] || cat "$stdout_file" >&2
  [[ ! -s "$stderr_file" ]] || cat "$stderr_file" >&2
  return 1
}

read_exact_sha() {
  local label=$1
  local response_file=$2
  local actual_sha

  actual_sha=$(tr -d '\r\n' <"$response_file")
  if [[ ! "$actual_sha" =~ ^[0-9a-f]{40}$ ]]; then
    echo "$label returned an invalid commit SHA: $actual_sha" >&2
    return 1
  fi
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "$label points to $actual_sha, expected $expected_sha" >&2
    return 1
  fi
}

tag_endpoint="repos/$repository/git/ref/tags/$tag"
if gh api "$tag_endpoint" --jq .object.sha \
  >"$lookup_stdout" 2>"$lookup_stderr"; then
  read_exact_sha "existing tag $tag" "$lookup_stdout"
  echo "verified existing tag $tag at $expected_sha"
  exit 0
else
  lookup_status=$?
fi
if ! jq -e -s \
  'length == 1
   and (.[0]
     | type == "object"
     and ((.status // "") | tostring) == "404")' \
  "$lookup_stdout" >/dev/null 2>&1; then
  fail_with_response \
    "failed to inspect tag $tag (gh exit $lookup_status; expected an explicit 404 for a missing tag)" \
    "$lookup_stdout" "$lookup_stderr"
fi

create_stdout="$scratch/create.stdout"
create_stderr="$scratch/create.stderr"
if ! gh api --method POST "repos/$repository/git/refs" \
  -f ref="refs/tags/$tag" \
  -f sha="$expected_sha" \
  --jq .object.sha >"$create_stdout" 2>"$create_stderr"; then
  fail_with_response "failed to create tag $tag" \
    "$create_stdout" "$create_stderr"
fi
read_exact_sha "created tag $tag" "$create_stdout"

reread_stdout="$scratch/reread.stdout"
reread_stderr="$scratch/reread.stderr"
if ! gh api "$tag_endpoint" --jq .object.sha \
  >"$reread_stdout" 2>"$reread_stderr"; then
  fail_with_response "failed to re-read created tag $tag" \
    "$reread_stdout" "$reread_stderr"
fi
read_exact_sha "re-read tag $tag" "$reread_stdout"
echo "created and verified tag $tag at $expected_sha"
