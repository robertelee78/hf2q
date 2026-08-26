#!/usr/bin/env bash
# Arm-aware physical-wave validation for the stable-boundary compound A/B.
# The exact main baseline may predate the physical batching counters. Both
# arms must complete four concurrent client requests; only the candidate is
# required to expose telemetry. This prefill/TTFT workload deliberately asks
# for a one-token semantic answer, so physical decode width is observed rather
# than required. The separate physical multi-slot matrix owns that proof.

qwen35_compound_wave_contract_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! declare -F qwen38_physical_metric_u64 >/dev/null; then
    # shellcheck source=scripts/qwen38_physical_multislot_contract.sh
    source "$qwen35_compound_wave_contract_dir/qwen38_physical_multislot_contract.sh"
fi

qwen35_compound_metric_set_state() {
    local before_path=$1 after_path=$2 path name matches present=0
    local metrics=(
        hf2q_qwen_decode_scheduler_steps_total
        hf2q_qwen_decode_scheduler_handles_total
        hf2q_qwen_decode_ordinary_target_forwards_total
        hf2q_qwen_decode_ordinary_target_body_rows_total
        hf2q_qwen_decode_ordinary_target_head_rows_total
        hf2q_qwen_decode_scheduler_max_width
        hf2q_qwen_decode_ordinary_target_body_max_width
        hf2q_qwen_decode_ordinary_target_head_max_width
    )

    for path in "$before_path" "$after_path"; do
        [[ -f "$path" && -r "$path" ]] || {
            echo "compound-wave metrics are missing or unreadable: $path" >&2
            return 1
        }
        for name in "${metrics[@]}"; do
            matches=$(awk -v name="$name" '$1 == name { count++ } END { print count + 0 }' "$path")
            (( matches <= 1 )) || {
                echo "duplicate compound-wave metric $name in $path" >&2
                return 1
            }
            present=$((present + matches))
        done
    done

    case "$present" in
        0) printf '%s\n' unavailable ;;
        16) printf '%s\n' available ;;
        *)
            echo "partial compound-wave telemetry: present=$present expected=0-or-16" >&2
            return 1
            ;;
    esac
}

prove_compound_wave() {
    local label=$1 trial=$2 before=$3 after=$4 output=$5
    local require_physical_instrumentation=$6 client_count=$7 instrumentation
    local scheduler_steps_before scheduler_steps_after scheduler_handles_before
    local scheduler_handles_after target_forwards_before target_forwards_after
    local body_rows_before body_rows_after head_rows_before head_rows_after
    local scheduler_steps_delta scheduler_handles_delta target_forwards_delta
    local body_rows_delta head_rows_delta scheduler_max body_max head_max
    local physical_width_four=false

    [[ "$label" =~ ^[a-z][a-z0-9_-]*$ ]] || {
        echo "invalid compound-wave label: $label" >&2
        return 1
    }
    [[ "$trial" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid compound-wave trial: $trial" >&2
        return 1
    }
    [[ "$require_physical_instrumentation" =~ ^[01]$ ]] || {
        echo "physical-instrumentation requirement must be 0 or 1" >&2
        return 1
    }
    [[ "$client_count" == 4 ]] || {
        echo "compound wave did not complete four clients: $client_count" >&2
        return 1
    }

    instrumentation=$(qwen35_compound_metric_set_state "$before" "$after") || return 1
    if [[ "$instrumentation" == unavailable ]]; then
        (( require_physical_instrumentation == 0 )) || {
            echo "$label trial $trial lacks required physical telemetry" >&2
            return 1
        }
        jq -n --arg label "$label" --argjson trial "$trial" '{
          label:$label,trial:$trial,client_wave_complete:true,client_count:4,
          physical_instrumentation:"unavailable",
          physical_width_four_observed:null,
          scheduler:null,target:null
        }' >"$output"
        return 0
    fi

    scheduler_steps_before=$(qwen38_physical_metric_u64 \
        "$before" hf2q_qwen_decode_scheduler_steps_total) || return 1
    scheduler_steps_after=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_scheduler_steps_total) || return 1
    scheduler_handles_before=$(qwen38_physical_metric_u64 \
        "$before" hf2q_qwen_decode_scheduler_handles_total) || return 1
    scheduler_handles_after=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_scheduler_handles_total) || return 1
    target_forwards_before=$(qwen38_physical_metric_u64 \
        "$before" hf2q_qwen_decode_ordinary_target_forwards_total) || return 1
    target_forwards_after=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_ordinary_target_forwards_total) || return 1
    body_rows_before=$(qwen38_physical_metric_u64 \
        "$before" hf2q_qwen_decode_ordinary_target_body_rows_total) || return 1
    body_rows_after=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_ordinary_target_body_rows_total) || return 1
    head_rows_before=$(qwen38_physical_metric_u64 \
        "$before" hf2q_qwen_decode_ordinary_target_head_rows_total) || return 1
    head_rows_after=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_ordinary_target_head_rows_total) || return 1
    scheduler_max=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_scheduler_max_width) || return 1
    body_max=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_ordinary_target_body_max_width) || return 1
    head_max=$(qwen38_physical_metric_u64 \
        "$after" hf2q_qwen_decode_ordinary_target_head_max_width) || return 1

    (( scheduler_steps_after >= scheduler_steps_before \
        && scheduler_handles_after >= scheduler_handles_before \
        && target_forwards_after >= target_forwards_before \
        && body_rows_after >= body_rows_before \
        && head_rows_after >= head_rows_before )) || {
        echo "$label trial $trial has a regressing physical counter" >&2
        return 1
    }
    scheduler_steps_delta=$((scheduler_steps_after - scheduler_steps_before))
    scheduler_handles_delta=$((scheduler_handles_after - scheduler_handles_before))
    target_forwards_delta=$((target_forwards_after - target_forwards_before))
    body_rows_delta=$((body_rows_after - body_rows_before))
    head_rows_delta=$((head_rows_after - head_rows_before))
    (( scheduler_steps_delta > 0 && scheduler_handles_delta > 0 \
        && target_forwards_delta > 0 && body_rows_delta > 0 \
        && head_rows_delta > 0 )) || {
        echo "$label trial $trial did not advance every physical counter" >&2
        return 1
    }
    if (( scheduler_max == 4 && body_max == 4 && head_max == 4 \
        && scheduler_handles_delta > 3 * scheduler_steps_delta \
        && body_rows_delta > 3 * target_forwards_delta \
        && head_rows_delta > 3 * target_forwards_delta )); then
        physical_width_four=true
    fi

    jq -n --arg label "$label" --argjson trial "$trial" \
        --argjson physical_width_four "$physical_width_four" \
        --argjson scheduler_steps_delta "$scheduler_steps_delta" \
        --argjson scheduler_handles_delta "$scheduler_handles_delta" \
        --argjson scheduler_max "$scheduler_max" \
        --argjson target_forwards_delta "$target_forwards_delta" \
        --argjson body_rows_delta "$body_rows_delta" \
        --argjson head_rows_delta "$head_rows_delta" \
        --argjson body_max "$body_max" --argjson head_max "$head_max" '{
          label:$label,trial:$trial,client_wave_complete:true,client_count:4,
          physical_instrumentation:"available",
          physical_width_four_observed:$physical_width_four,
          scheduler:{steps_delta:$scheduler_steps_delta,
            handles_delta:$scheduler_handles_delta,max_width:$scheduler_max},
          target:{forwards_delta:$target_forwards_delta,
            body_rows_delta:$body_rows_delta,head_rows_delta:$head_rows_delta,
            body_max_width:$body_max,head_max_width:$head_max}
        }' >"$output"

}

qwen35_compound_validate_policy() {
    local single_ttft=$1 single_wall=$2 four_slot_wave=$3
    [[ "$single_ttft" == 1.01 && "$single_wall" == 1.0 \
        && "$four_slot_wave" == 1.0 ]] || {
        echo "noncanonical compound A/B acceptance policy: ttft=$single_ttft single_wall=$single_wall four_slot_wave=$four_slot_wave" >&2
        return 1
    }
}

qwen35_compound_require_fresh_out_dir() {
    local output=$1
    [[ "$output" == /* && ! -L "$output" ]] || {
        echo "compound A/B OUT_DIR must be absolute and non-symlink: $output" >&2
        return 1
    }
    if [[ -e "$output" ]]; then
        [[ -d "$output" ]] || {
            echo "compound A/B OUT_DIR is not a directory: $output" >&2
            return 1
        }
        [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
            echo "compound A/B OUT_DIR is not fresh and empty: $output" >&2
            return 1
        }
    else
        mkdir -p "$output"
    fi
}

qwen35_compound_publish_receipt() {
    local temporary=$1 final=$2
    [[ -f "$temporary" && ! -L "$temporary" && ! -e "$final" ]] || {
        echo "compound A/B receipt publication requires a new final path" >&2
        return 1
    }
    jq -e '.schema == 3 and .verdict == "pass"' "$temporary" >/dev/null || {
        echo "compound A/B temporary receipt is not a schema-3 pass" >&2
        return 1
    }
    mv "$temporary" "$final"
}

qwen35_compound_aggregate_arm() {
    local arm=$1 first=$2 second=$3 output=$4
    jq -s --arg arm "$arm" '
      def median:
        sort as $s | ($s | length) as $n
        | if $n == 0 then error("empty timing samples")
          elif ($n % 2) == 1 then $s[($n / 2 | floor)]
          else (($s[$n / 2 - 1] + $s[$n / 2]) / 2)
          end;
      .[0] as $first
      | if length != 2 then error("ABBA arm requires two process summaries")
        elif any(.[];
          .binary != $first.binary
          or .binary_sha256 != $first.binary_sha256
          or .source_commit != $first.source_commit
          or .dependency_identity != $first.dependency_identity
          or .model_id != $first.model_id)
        then error("ABBA arm process identity drift")
        else
          ([.[].single_ttft_samples_ms[]]) as $ttft
          | ([.[].single_wall_samples_seconds[]]) as $single_wall
          | ([.[].four_slot_wave_samples_seconds[]]) as $wave
          | {
              label:$arm,
              binary:$first.binary,
              binary_sha256:$first.binary_sha256,
              source_commit:$first.source_commit,
              dependency_identity:$first.dependency_identity,
              model_id:$first.model_id,
              process_order:[.[].label],
              process_summaries:.,
              compound_receipts:(map(.compound_receipts) | add),
              single_ttft_samples_ms:$ttft,
              single_wall_samples_seconds:$single_wall,
              four_slot_wave_samples_seconds:$wave,
              single_median_ttft_ms:($ttft | median),
              single_median_wall_seconds:($single_wall | median),
              four_slot_median_wave_seconds:($wave | median),
              wave_execution_receipts:[.[].wave_execution_receipts[]]
            }
        end
    ' "$first" "$second" >"$output"
}
