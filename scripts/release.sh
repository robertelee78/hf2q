#!/usr/bin/env bash
# Operator command: dispatch the Release workflow and see it through hosted
# macOS capacity losses.
#
#   scripts/release.sh VERSION SHA        dispatch release.yml and watch it
#   scripts/release.sh --resume RUN_ID    keep retrying an existing Release run
#
# When a job is cancelled because GitHub never acquired a hosted runner (the
# "not acquired by Runner" annotation), this re-runs the failed jobs of the
# SAME run (`gh run rerun --failed`), so the run keeps its candidate artifacts
# and the draft resumes with identical bytes. Any other failure stops at once
# and prints the failing job and step. Not a workflow: run it locally with an
# authenticated gh.
#
# Environment:
#   HF2Q_RELEASE_MAX_RERUNS   capacity reruns before giving up (default 6)
#   HF2Q_RELEASE_POLL_SECONDS run status poll interval (default 30)
set -euo pipefail

max_reruns=${HF2Q_RELEASE_MAX_RERUNS:-6}
poll_seconds=${HF2Q_RELEASE_POLL_SECONDS:-30}
capacity_marker='not acquired by Runner'

usage() {
  echo "usage: $0 VERSION SHA | $0 --resume RUN_ID" >&2
  exit 2
}

log() {
  printf '[release %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2
}

for command in gh jq; do
  command -v "$command" >/dev/null || { echo "$command is required" >&2; exit 1; }
done
[[ "$max_reruns" =~ ^[0-9]+$ && "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || usage

version=""
run_id=""
case "${1:-}" in
  --resume)
    [[ $# -eq 2 && "$2" =~ ^[1-9][0-9]*$ ]] || usage
    run_id=$2
    ;;
  *)
    [[ $# -eq 2 ]] || usage
    version=$1
    sha=$2
    [[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || {
      echo "VERSION must be canonical stable SemVer" >&2
      exit 2
    }
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || {
      echo "SHA must be a full 40-character commit SHA" >&2
      exit 2
    }
    ;;
esac

repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

if [[ -z "$run_id" ]]; then
  dispatched_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  login=$(gh api user --jq .login)
  gh workflow run release.yml --repo "$repo" \
    -f commit_sha="$sha" -f version="$version"
  log "dispatched release.yml for hf2q $version at $sha; locating the run"
  for _ in $(seq 1 30); do
    run_id=$(gh run list --repo "$repo" --workflow release.yml \
      --event workflow_dispatch --user "$login" --limit 10 \
      --json databaseId,createdAt \
      --jq "[.[] | select(.createdAt >= \"$dispatched_at\")] | sort_by(.createdAt) | last | .databaseId // empty")
    [[ -n "$run_id" ]] && break
    sleep 5
  done
  [[ -n "$run_id" ]] || { echo "dispatched run did not appear" >&2; exit 1; }
fi

run_json=$(gh run view "$run_id" --repo "$repo" --json workflowName,url)
[[ $(jq -r .workflowName <<<"$run_json") == Release ]] || {
  echo "run $run_id is not a Release run" >&2
  exit 1
}
log "watching $(jq -r .url <<<"$run_json")"

# Print every non-successful, non-skipped job of the latest attempt as
# "<job id>\t<conclusion>\t<name>".
unsuccessful_jobs() {
  local attempt=$1
  gh api --paginate \
    "repos/$repo/actions/runs/$run_id/attempts/$attempt/jobs?per_page=100" \
    --jq '.jobs[] | select(.conclusion != "success" and .conclusion != "skipped" and .conclusion != null) | [.id, .conclusion, .name] | @tsv'
}

job_lost_runner() {
  local job_id=$1
  # A job's check-run id equals its job id.
  gh api "repos/$repo/check-runs/$job_id/annotations" --jq '.[].message' |
    grep -Fq "$capacity_marker"
}

print_failure() {
  local job_id=$1 name=$2
  local step
  step=$(gh api "repos/$repo/actions/jobs/$job_id" \
    --jq '[.steps[]? | select(.conclusion == "failure")][0].name // "(no failed step recorded)"')
  echo "FAILED job: $name" >&2
  echo "FAILED step: $step" >&2
  gh api "repos/$repo/check-runs/$job_id/annotations" \
    --jq '.[] | "  \(.annotation_level): \(.message)"' >&2 || true
  echo "log: gh run view $run_id --repo $repo --job $job_id --log-failed" >&2
}

reruns=0
while true; do
  while true; do
    status_json=$(gh run view "$run_id" --repo "$repo" \
      --json status,conclusion,attempt)
    [[ $(jq -r .status <<<"$status_json") == completed ]] && break
    sleep "$poll_seconds"
  done
  conclusion=$(jq -r .conclusion <<<"$status_json")
  attempt=$(jq -r .attempt <<<"$status_json")
  if [[ "$conclusion" == success ]]; then
    break
  fi

  capacity_losses=0
  real_failures=0
  while IFS=$'\t' read -r job_id job_conclusion job_name; do
    [[ -n "$job_id" ]] || continue
    if job_lost_runner "$job_id"; then
      log "attempt $attempt: '$job_name' never got a hosted runner ($job_conclusion)"
      ((capacity_losses += 1))
    else
      print_failure "$job_id" "$job_name"
      ((real_failures += 1))
    fi
  done < <(unsuccessful_jobs "$attempt")

  if ((real_failures > 0)); then
    echo "release run $run_id attempt $attempt failed for a non-capacity reason; not retrying" >&2
    exit 1
  fi
  if ((capacity_losses == 0)); then
    echo "release run $run_id attempt $attempt concluded '$conclusion' with no failed job to classify; not retrying" >&2
    exit 1
  fi
  if ((reruns >= max_reruns)); then
    echo "hosted macOS capacity still unavailable after $reruns reruns; resume later with: $0 --resume $run_id" >&2
    exit 1
  fi
  delay=$((60 * (1 << reruns)))
  if ((delay > 900)); then delay=900; fi
  ((reruns += 1))
  log "capacity loss; rerunning failed jobs of the same run in ${delay}s (rerun $reruns/$max_reruns)"
  sleep "$delay"
  gh run rerun "$run_id" --repo "$repo" --failed
  # Do not read the finished attempt's status as the new attempt's.
  until (($(gh run view "$run_id" --repo "$repo" --json attempt --jq .attempt) > attempt)); do
    sleep 5
  done
done

if [[ -n "$version" ]]; then
  release_url=$(gh release view "v$version" --repo "$repo" --json url --jq .url)
else
  release_url=$(gh release view --repo "$repo" --json url --jq .url)
fi
log "release run $run_id succeeded"
echo "$release_url"
