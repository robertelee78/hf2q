#!/usr/bin/env bash
# Live, model-family-neutral gate for the r2c Stage 6 ReviewLens and Stage 9
# CWE response schemas. The server must already be running; this script never
# loads or stops a model.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
MODEL="${MODEL:-}"
MAX_TOKENS="${MAX_TOKENS:-768}"
KEEP_WORK_DIR="${KEEP_WORK_DIR:-0}"

for tool in curl jq; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "required tool not found: $tool" >&2
        exit 2
    }
done
if ! [[ "$BASE_URL" =~ ^http://(127\.0\.0\.1|localhost):[0-9]+$ ]]; then
    echo "BASE_URL must be a loopback endpoint without /v1" >&2
    exit 2
fi
if ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ && "$KEEP_WORK_DIR" =~ ^[01]$ ]]; then
    echo "MAX_TOKENS must be positive and KEEP_WORK_DIR must be 0 or 1" >&2
    exit 2
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-r2c-structured.XXXXXX")
cleanup() {
    local status=$?
    if (( KEEP_WORK_DIR == 1 || status != 0 )); then
        echo "r2c structured-output workspace retained at $work_dir" >&2
    else
        rm -rf "$work_dir"
    fi
}
trap cleanup EXIT

if [[ -z "$MODEL" ]]; then
    MODEL=$(curl --fail-with-body --silent --show-error "$BASE_URL/v1/models" |
        jq -er '.data[0].id')
fi

stage6_schema="$REPO_ROOT/tests/fixtures/structured_output/r2c/stage6_review_lens.schema.json"
stage9_schema="$REPO_ROOT/tests/fixtures/structured_output/r2c/stage9_cwe.schema.json"

build_request() {
    local stage=$1
    local schema=$2
    local output=$3
    local prompt
    if [[ "$stage" == "stage6_review_lens" ]]; then
        prompt='Review unit U1 through lens L1. No members, relationships, facts, gaps, or observations are present. Emit schema_version 1 and disposition examined_no_material_observation.'
    else
        prompt='Classify evidence M1 as CWE-79 with no related CWEs, rationale input is rendered without escaping, and evidence_refs containing M1.'
    fi
    jq -n \
        --arg model "$MODEL" \
        --arg stage "$stage" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        --slurpfile schema "$schema" '{
          model: $model,
          messages: [
            {
              role: "system",
              content: "Return only concise JSON conforming exactly to the supplied response schema."
            },
            {role: "user", content: $prompt}
          ],
          response_format: {
            type: "json_schema",
            json_schema: {name: $stage, strict: true, schema: $schema[0]}
          },
          temperature: 0,
          max_tokens: $max_tokens,
          stream: false
        }' >"$output"
}

validate_stage6() {
    jq -e '
      type == "object"
      and .schema_version == 1
      and .review_unit_alias == "U1"
      and .lens_alias == "L1"
      and .disposition == "examined_no_material_observation"
      and (.member_coverage | type == "array" and length == 0)
      and (.relationship_coverage | type == "array" and length == 0)
      and (.fact_coverage | type == "array" and length == 0)
      and (.gap_coverage | type == "array" and length == 0)
      and (.observations | type == "array" and length == 0)
      and ((keys | sort) == ([
        "schema_version", "review_unit_alias", "lens_alias", "disposition",
        "member_coverage", "relationship_coverage", "fact_coverage",
        "gap_coverage", "observations"
      ] | sort))
    ' "$1" >/dev/null
}

validate_stage9() {
    jq -e '
      type == "object"
      and .schema_version == 1
      and .primary_cwe == "CWE-79"
      and (.related_cwes | type == "array" and length == 0)
      and (.rationale | type == "string" and length > 0)
      and .evidence_refs == ["M1"]
      and ((keys | sort) == ([
        "schema_version", "primary_cwe", "related_cwes", "rationale",
        "evidence_refs"
      ] | sort))
    ' "$1" >/dev/null
}

post_unary() {
    local stage=$1
    local request=$2
    local response="$work_dir/$stage-response.json"
    local content="$work_dir/$stage-content.json"
    curl --fail-with-body --silent --show-error \
        -H 'content-type: application/json' \
        --data-binary "@$request" \
        "$BASE_URL/v1/chat/completions" >"$response"
    jq -er '.choices[0].finish_reason == "stop" and (.choices[0].message.content | type == "string")' \
        "$response" >/dev/null
    jq -er '.choices[0].message.content' "$response" >"$content"
    "validate_$stage" "$content"
}

stage6_request="$work_dir/stage6-request.json"
stage9_request="$work_dir/stage9-request.json"
build_request stage6_review_lens "$stage6_schema" "$stage6_request"
build_request stage9_cwe "$stage9_schema" "$stage9_request"
post_unary stage6 "$stage6_request"
post_unary stage9 "$stage9_request"

# Exercise response-format grammar through SSE independently of the native
# tool-call SSE gate in test_deepseek4_structured_tools.sh.
stage9_sse="$work_dir/stage9-response.sse"
stage9_jsonl="$work_dir/stage9-response.jsonl"
stage9_streamed="$work_dir/stage9-streamed-content.json"
jq '.stream = true | .stream_options = {include_usage: true}' "$stage9_request" |
    curl --fail-with-body --silent --show-error --no-buffer \
        -H 'content-type: application/json' \
        --data-binary @- "$BASE_URL/v1/chat/completions" >"$stage9_sse"
if [[ $(grep -c '^data: \[DONE\]$' "$stage9_sse" || true) != 1 ]]; then
    echo "r2c structured-output gate failed: SSE did not terminate exactly once" >&2
    exit 1
fi
sed -n 's/^data: //p' "$stage9_sse" | grep -v '^\[DONE\]$' >"$stage9_jsonl"
jq -jr -s '[.[] | .choices[]?.delta.content? // empty] | join("")' \
    "$stage9_jsonl" >"$stage9_streamed"
validate_stage9 "$stage9_streamed"

jq -n \
    --arg model "$MODEL" \
    --argjson max_tokens "$MAX_TOKENS" '{
      status: "pass",
      model: $model,
      max_tokens: $max_tokens,
      stage6_review_lens_unary_valid: true,
      stage9_cwe_unary_valid: true,
      stage9_cwe_sse_valid: true
    }'
