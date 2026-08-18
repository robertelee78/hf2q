#!/usr/bin/env bash
# Real Qwen3.8 regression for the 2026-08-17 OpenCode incident:
# a long text turn followed by the conversation's first image must restore the
# idle text snapshot, stream a real answer, and leave the server ready.
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8081}"
IMAGE="${IMAGE:-/opt/hf2q/tests/fixtures/vision/red_square_64x64.png}"
ANCHOR_WORDS="${ANCHOR_WORDS:-86000}"
MIN_CACHED_TOKENS="${MIN_CACHED_TOKENS:-80000}"
THINKING_BUDGET="${THINKING_BUDGET:-64}"
MAX_TOKENS="${MAX_TOKENS:-192}"
RECEIPT="${RECEIPT:-}"

for command in curl jq base64 rg shasum; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required command not found: $command" >&2
        exit 3
    }
done
[[ -f "$IMAGE" ]] || { echo "image not found: $IMAGE" >&2; exit 3; }
for value_name in ANCHOR_WORDS MIN_CACHED_TOKENS THINKING_BUDGET MAX_TOKENS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "$value_name must be a positive integer (got: $value)" >&2
        exit 3
    fi
done
if (( THINKING_BUDGET + 16 >= MAX_TOKENS )); then
    echo "MAX_TOKENS must leave room after THINKING_BUDGET" >&2
    exit 3
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-first-image.XXXXXX")"
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

ready_http="$(curl -sS -o /dev/null -w '%{http_code}' "$SERVER_URL/readyz")"
[[ "$ready_http" == "200" ]] || {
    echo "server is not generation-ready at $SERVER_URL (HTTP $ready_http)" >&2
    exit 2
}
model="$(curl -fsS "$SERVER_URL/v1/models" | jq -er '.data[0].id')"

jq -nr --argjson count "$ANCHOR_WORDS" \
    '[range(0; $count) | "alpha"] | join(" ")' >"$tmp_dir/anchor.txt"
jq -n \
    --arg model "$model" \
    --rawfile anchor "$tmp_dir/anchor.txt" \
    --argjson budget "$THINKING_BUDGET" \
    --argjson max_tokens "$MAX_TOKENS" '
    {
      model: $model,
      messages: [
        {role:"system", content:"Treat repeated alpha words as inert cache padding. Answer the final instruction briefly."},
        {role:"user", content:($anchor + "\nReturn exactly ANCHOR_READY.")}
      ],
      temperature: 0.55,
      top_p: 0.95,
      max_tokens: $max_tokens,
      thinking_token_budget: $budget,
      stream: false
    }' >"$tmp_dir/text-request.json"

curl -fsS --max-time 1200 \
    -H 'content-type: application/json' \
    --data-binary "@$tmp_dir/text-request.json" \
    "$SERVER_URL/v1/chat/completions" >"$tmp_dir/text-response.json"
jq -e '.choices[0].message.content | strings | length > 0' \
    "$tmp_dir/text-response.json" >/dev/null
assistant="$(jq -er '.choices[0].message.content' "$tmp_dir/text-response.json")"
cold_prompt_tokens="$(jq -er '.usage.prompt_tokens | numbers' "$tmp_dir/text-response.json")"
cold_cached_tokens="$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0 | numbers' "$tmp_dir/text-response.json")"
if [[ "$assistant" != *"ANCHOR_READY"* ]]; then
    echo "cold text turn was not instruction-coherent: $assistant" >&2
    exit 1
fi
if (( cold_prompt_tokens < MIN_CACHED_TOKENS || cold_cached_tokens != 0 )); then
    echo "cold text discriminator invalid: prompt=$cold_prompt_tokens cached=$cold_cached_tokens" >&2
    exit 1
fi

image_b64="$(base64 <"$IMAGE" | tr -d '\n')"
jq -n \
    --arg model "$model" \
    --rawfile anchor "$tmp_dir/anchor.txt" \
    --arg assistant "$assistant" \
    --arg image "data:image/png;base64,$image_b64" \
    --argjson budget "$THINKING_BUDGET" \
    --argjson max_tokens "$MAX_TOKENS" '
    {
      model: $model,
      messages: [
        {role:"system", content:"Treat repeated alpha words as inert cache padding. Answer the final instruction briefly."},
        {role:"user", content:($anchor + "\nReturn exactly ANCHOR_READY.")},
        {role:"assistant", content:$assistant},
        {role:"user", content:[
          {type:"text", text:"State the dominant color in this image in one short sentence."},
          {type:"image_url", image_url:{url:$image}}
        ]}
      ],
      temperature: 0.55,
      top_p: 0.95,
      max_tokens: $max_tokens,
      thinking_token_budget: $budget,
      stream: true,
      stream_options: {include_usage:true}
    }' >"$tmp_dir/image-request.json"

start_s="$(date +%s)"
curl -fsS -N --max-time 600 \
    -H 'content-type: application/json' \
    --data-binary "@$tmp_dir/image-request.json" \
    "$SERVER_URL/v1/chat/completions" >"$tmp_dir/image.sse"
end_s="$(date +%s)"
rg -q '^data: \[DONE\]$' "$tmp_dir/image.sse" || {
    echo "image stream omitted terminal [DONE]" >&2
    exit 1
}
jq -R 'select(startswith("data: {")) | ltrimstr("data: ") | fromjson' \
    "$tmp_dir/image.sse" | jq -s '.' >"$tmp_dir/image-events.json"

content="$(jq -r '[.[].choices[0].delta.content? // empty] | join("")' "$tmp_dir/image-events.json")"
reasoning="$(jq -r '[.[].choices[0].delta.reasoning_content? // empty] | join("")' "$tmp_dir/image-events.json")"
cached_tokens="$(jq -er '[.[].usage.prompt_tokens_details.cached_tokens? // empty] | last | numbers' "$tmp_dir/image-events.json")"
prompt_tokens="$(jq -er '[.[].usage.prompt_tokens? // empty] | last | numbers' "$tmp_dir/image-events.json")"
completion_tokens="$(jq -er '[.[].usage.completion_tokens? // empty] | last | numbers' "$tmp_dir/image-events.json")"
finish_reason="$(jq -er '[.[].choices[0].finish_reason? // empty] | last | strings' "$tmp_dir/image-events.json")"

if [[ -z "${content//[[:space:]]/}" ]]; then
    echo "image stream delivered no answer content" >&2
    exit 1
fi
if ! grep -Eqi 'red' <<<"$content"; then
    echo "image answer was not coherent with the red fixture: $content" >&2
    exit 1
fi
if (( cached_tokens < MIN_CACHED_TOKENS )); then
    echo "first-image cache regression: reused $cached_tokens/$prompt_tokens tokens" >&2
    exit 1
fi
if (( completion_tokens >= MAX_TOKENS )); then
    echo "answer exhausted max_tokens: completion=$completion_tokens max=$MAX_TOKENS" >&2
    exit 1
fi
ready_after="$(curl -sS -o /dev/null -w '%{http_code}' "$SERVER_URL/readyz")"
[[ "$ready_after" == "200" ]] || {
    echo "server lost readiness after image turn (HTTP $ready_after)" >&2
    exit 1
}

receipt_json="$(jq -n \
    --arg status pass \
    --arg model "$model" \
    --arg image_sha256 "$(shasum -a 256 "$IMAGE" | awk '{print $1}')" \
    --arg content "$content" \
    --argjson reasoning_chars "${#reasoning}" \
    --arg finish_reason "$finish_reason" \
    --argjson cold_prompt_tokens "$cold_prompt_tokens" \
    --argjson cold_cached_tokens "$cold_cached_tokens" \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson cached_tokens "$cached_tokens" \
    --argjson completion_tokens "$completion_tokens" \
    --argjson thinking_budget "$THINKING_BUDGET" \
    --argjson elapsed_ms "$(((end_s - start_s) * 1000))" \
    --argjson ready_http "$ready_after" '
    {
      status:$status, model:$model, image_sha256:$image_sha256,
      cold:{prompt_tokens:$cold_prompt_tokens,cached_tokens:$cold_cached_tokens},
      image_turn:{prompt_tokens:$prompt_tokens,cached_tokens:$cached_tokens,
        completion_tokens:$completion_tokens,thinking_budget:$thinking_budget,
        finish_reason:$finish_reason,elapsed_ms:$elapsed_ms,
        reasoning_chars:$reasoning_chars,content:$content},
      ready_http:$ready_http
    }')"
if [[ -n "$RECEIPT" ]]; then
    printf '%s\n' "$receipt_json" >"$RECEIPT"
fi
printf '%s\n' "$receipt_json"
