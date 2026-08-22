#!/usr/bin/env bash
# Shared fail-closed validators for the Qwen3.8 physical multi-slot gate.
# This file is sourced by the real-server runner and model-free tests.

qwen38_physical_metric_u64() {
    local metrics_path=$1
    local metric_name=$2
    local value

    [[ -f "$metrics_path" && -r "$metrics_path" ]] || {
        echo "physical-width metrics are missing or unreadable: $metrics_path" >&2
        return 1
    }
    value=$(awk -v name="$metric_name" '
        $1 == name {
            if (++matches != 1 || NF != 2 || $2 !~ /^[0-9]+$/) exit 2
            value = $2
        }
        END {
            if (matches != 1) exit 3
            print value
        }
    ' "$metrics_path") || {
        echo "expected one integer metric named $metric_name in $metrics_path" >&2
        return 1
    }
    printf '%s\n' "$value"
}

qwen38_physical_validate_response() {
    local response_path=$1
    local expected_model=$2

    jq -e --arg model "$expected_model" '
        . as $response
        | .object == "chat.completion"
        and .model == $model
        and (.choices | type == "array" and length == 1)
        and .choices[0].index == 0
        and .choices[0].message.role == "assistant"
        and (.choices[0].message.content | type == "string" and length > 0)
        and (.choices[0].finish_reason | type == "string" and length > 0)
        and (.usage.prompt_tokens | type == "number" and . > 0)
        and (.usage.completion_tokens | type == "number" and . > 0)
        and (.usage.total_tokens | type == "number"
             and . == ($response.usage.prompt_tokens
                       + $response.usage.completion_tokens))
        and (.usage.prompt_tokens_details.cached_tokens
             | type == "number" and . >= 0)
        and (.x_hf2q_timing.decode_time_secs | type == "number" and . > 0)
        and (.x_hf2q_timing.total_time_secs | type == "number" and . > 0)
        and (.x_hf2q_timing.time_to_first_token_ms
             | type == "number" and . > 0)
        and (.x_hf2q_timing.decode_tokens_per_sec
             | type == "number" and . > 0)
    ' "$response_path" >/dev/null || {
        echo "invalid unary Qwen3.8 response: $response_path" >&2
        return 1
    }
}

qwen38_physical_validate_equal_prompt_tokens() {
    local expected_count=$1
    shift
    local counts

    [[ "$expected_count" =~ ^[0-9]+$ && "$expected_count" -gt 0 ]] || {
        echo "expected response count must be a positive integer" >&2
        return 1
    }
    [[ "$#" -eq "$expected_count" ]] || {
        echo "response cardinality drift: expected=$expected_count actual=$#" >&2
        return 1
    }
    counts=$(jq -er '.usage.prompt_tokens' "$@" | sort -nu) || return 1
    [[ "$(printf '%s\n' "$counts" | awk 'NF { count++ } END { print count + 0 }')" == 1 ]] || {
        echo "physical-width requests did not tokenize to an equal prompt length" >&2
        printf '%s\n' "$counts" >&2
        return 1
    }
}

qwen38_physical_validate_metrics() {
    local width=$1
    local before_path=$2
    local after_path=$3
    local name before after delta
    local scheduler_max body_max head_max
    local counters=(
        hf2q_qwen_decode_scheduler_steps_total
        hf2q_qwen_decode_scheduler_handles_total
        hf2q_qwen_decode_ordinary_target_forwards_total
        hf2q_qwen_decode_ordinary_target_body_rows_total
        hf2q_qwen_decode_ordinary_target_head_rows_total
        hf2q_qwen_decode_ordinary_command_buffer_submissions_total
    )
    local monotonic_counters=(
        hf2q_qwen_decode_ordinary_command_buffers_created_total
    )

    [[ "$width" =~ ^(1|2|4|8|16)$ ]] || {
        echo "physical width must be one of 1, 2, 4, 8, or 16" >&2
        return 1
    }

    scheduler_max=$(qwen38_physical_metric_u64 \
        "$after_path" hf2q_qwen_decode_scheduler_max_width) || return 1
    body_max=$(qwen38_physical_metric_u64 \
        "$after_path" hf2q_qwen_decode_ordinary_target_body_max_width) || return 1
    head_max=$(qwen38_physical_metric_u64 \
        "$after_path" hf2q_qwen_decode_ordinary_target_head_max_width) || return 1
    if [[ "$scheduler_max" != "$width" || "$body_max" != "$width" \
        || "$head_max" != "$width" ]]; then
        echo "physical width proof failed: requested=$width scheduler=$scheduler_max body=$body_max head=$head_max" >&2
        return 1
    fi

    for name in "${counters[@]}"; do
        before=$(qwen38_physical_metric_u64 "$before_path" "$name") || return 1
        after=$(qwen38_physical_metric_u64 "$after_path" "$name") || return 1
        (( after >= before )) || {
            echo "physical-width counter regressed: $name before=$before after=$after" >&2
            return 1
        }
        delta=$((after - before))
        (( delta > 0 )) || {
            echo "physical-width wave did not advance $name" >&2
            return 1
        }
    done
    for name in "${monotonic_counters[@]}"; do
        before=$(qwen38_physical_metric_u64 "$before_path" "$name") || return 1
        after=$(qwen38_physical_metric_u64 "$after_path" "$name") || return 1
        (( after >= before )) || {
            echo "physical-width counter regressed: $name before=$before after=$after" >&2
            return 1
        }
    done
}
