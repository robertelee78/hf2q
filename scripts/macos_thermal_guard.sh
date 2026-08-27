#!/usr/bin/env bash

# Shared macOS thermal-state helpers for calibrated hardware gates. This file
# is sourced by release scripts and intentionally performs no work on import.

_HF2Q_THERMAL_GUARD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
THERMAL_PROBE_BIN=""
THERMAL_PROBE_OWNED_DIR=""
THERMAL_PROBE_SOURCE=""
THERMAL_PROBE_COMPILER=""
THERMAL_PROBE_COMPILER_VERSION=""
THERMAL_SAMPLED_AT=""
# This sourced global is consumed by receipt producers.
# shellcheck disable=SC2034
HOST_CONTENTION_POLICY="process-group-cpu-v2"
HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT="100.0"
HOST_CONTENTION_STATE=""
HOST_CONTENTION_OWNER_PGID=""
HOST_CONTENTION_FOREIGN_CPU_PERCENT=""
HOST_CONTENTION_OFFENDERS=""

# Emit a normalized process snapshot for the calibrated host-contention guard.
# Process-group ownership is already established by the release workflow's
# process-group supervisor; no foreign process is ever signaled by this helper.
host_contention_process_snapshot() {
  local snapshot

  snapshot=$(/bin/ps -axo pid=,pgid=,%cpu=,command= 2>/dev/null | awk '
    {
      pid = $1
      pgid = $2
      cpu = $3
      $1 = ""
      $2 = ""
      $3 = ""
      sub(/^[[:space:]]+/, "", $0)
      if (pid !~ /^[1-9][0-9]*$/ || pgid !~ /^[1-9][0-9]*$/ \
          || cpu !~ /^[0-9]+([.][0-9]+)?$/ || length($0) == 0) {
        exit 2
      }
      gsub(/\t/, " ", $0)
      printf "%s\t%s\t%s\t%s\n", pid, pgid, cpu, $0
    }
  ') || {
    echo "failed to read a normalized host process snapshot" >&2
    return 1
  }
  [[ -n "$snapshot" ]] || {
    echo "host process snapshot was empty" >&2
    return 1
  }
  printf '%s\n' "$snapshot"
}

# Prove that a calibrated leaf owns a dedicated process group. Merely finding
# the caller in some inherited PGID is insufficient because unrelated work in
# that group would otherwise be excluded from foreign-CPU accounting.
host_contention_require_isolated_gate_owner() {
  local owner_pid=$1
  local snapshot

  [[ "$owner_pid" =~ ^[1-9][0-9]*$ ]] || return 2
  snapshot=$(host_contention_process_snapshot) || return 1
  awk -F '\t' -v owner="$owner_pid" '
    BEGIN { invalid = 0; count = 0 }
    {
      if (NF != 4 || $1 !~ /^[1-9][0-9]*$/ \
          || $2 !~ /^[1-9][0-9]*$/ \
          || $3 !~ /^[0-9]+([.][0-9]+)?$/ || length($4) == 0 \
          || seen[$1]++) {
        invalid = 1
        next
      }
      count++
      pgid_by_pid[$1] = $2
      command_by_pid[$1] = $4
    }
    END {
      if (invalid || count == 0 || !(owner in pgid_by_pid) \
          || pgid_by_pid[owner] != owner) exit 1
      for (pid in pgid_by_pid) {
        if (pid != owner && pgid_by_pid[pid] == owner) {
          name = command_by_pid[pid]
          sub(/[[:space:]].*$/, "", name)
          sub(/^.*\//, "", name)
          # The snapshot itself briefly creates these descendants inside the
          # fresh group. Any workload process is still a hard failure.
          if (name !~ /^(bash|ps|awk)$/) foreign_members++
        }
      }
      if (foreign_members != 0) exit 1
    }
  ' <<<"$snapshot" || {
    echo "calibrated leaf does not own an isolated process group: $owner_pid" >&2
    return 1
  }
}

host_contention_sample() {
  local log_file=$1
  local phase=$2
  local owner_pid=$3
  local sampled_at=${4:-}
  local owned_server_pid=${5:-}
  local snapshot
  local classification

  # Matched-engine gates may exempt exactly their verified hf2q or
  # llama-server PID. The PID must exist in the owner's process group; every
  # other compiler/model process and all foreign CPU remain fail-closed.

  [[ "$owner_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "host contention owner pid must be a positive integer" >&2
    return 2
  }
  if [[ -n "$owned_server_pid" && ! "$owned_server_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "owned server pid must be a positive integer" >&2
    return 2
  fi
  if [[ -z "$sampled_at" ]]; then
    sampled_at=$(date +%s) || return 1
  fi
  [[ "$sampled_at" =~ ^[0-9]+$ ]] || {
    echo "host contention sample timestamp must be a non-negative integer" >&2
    return 2
  }
  snapshot=$(host_contention_process_snapshot) || return 1
  classification=$(awk -F '\t' -v owner="$owner_pid" \
    -v allowed_owned="$owned_server_pid" \
    -v maximum="$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" '
    BEGIN { invalid = 0; count = 0 }
    {
      if (NF != 4 || $1 !~ /^[1-9][0-9]*$/ \
          || $2 !~ /^[1-9][0-9]*$/ \
          || $3 !~ /^[0-9]+([.][0-9]+)?$/ || length($4) == 0 \
          || seen[$1]++) {
        invalid = 1
        next
      }
      count++
      pid[count] = $1
      pgid[count] = $2
      pgid_by_pid[$1] = $2
      cpu[count] = $3 + 0
      command[count] = $4
      command_by_pid[$1] = $4
    }
    END {
      if (invalid || count == 0 || !(owner in pgid_by_pid) \
          || (allowed_owned != "" \
            && (!(allowed_owned in pgid_by_pid) \
              || pgid_by_pid[allowed_owned] != pgid_by_pid[owner])) \
          || maximum !~ /^[0-9]+([.][0-9]+)?$/ || maximum + 0 <= 0) exit 2
      owner_pgid = pgid_by_pid[owner]
      if (allowed_owned != "") {
        allowed_name = command_by_pid[allowed_owned]
        sub(/[[:space:]].*$/, "", allowed_name)
        sub(/^.*\//, "", allowed_name)
        if (allowed_name !~ /^(hf2q|llama-server)(-|$)/) exit 2
      }
      offenders = ""
      foreign_cpu = 0
      for (i = 1; i <= count; i++) {
        full_command = tolower(command[i])
        name = full_command
        sub(/[[:space:]].*$/, "", name)
        sub(/^.*\//, "", name)
        if (pgid[i] != owner_pgid) foreign_cpu += cpu[i]
        owned_server = allowed_owned != "" && pid[i] == allowed_owned \
          && pgid[i] == owner_pgid \
          && name ~ /^(hf2q|llama-server)(-|$)/
        python_model_work = name ~ /^python(3([.][0-9]+)?)?$/ \
          && full_command ~ /(mlx|torch|transformers|teacher|model[-_ ]?gen|inference|vllm)/
        forbidden = !owned_server \
          && ((name ~ /^(cargo|rustc|llama-cli|llama-server|llama-bench|ollama|mlx-lm|mlx_lm|swift-frontend)([0-9.-]|$)/) \
            || (name ~ /^hf2q([0-9.-]|$)/ && pgid[i] != owner_pgid) \
            || python_model_work)
        if (forbidden) {
          label = python_model_work ? "python-model-work" : name
          gsub(/[^A-Za-z0-9._+-]/, "_", label)
          item = pid[i] ":" pgid[i] ":" label
          offenders = offenders == "" ? item : offenders "," item
        }
      }
      state = (offenders != "" || foreign_cpu >= maximum + 0) ? "contended" : "quiet"
      printf "%s\t%.1f\t%s\t%s\n", owner_pgid, foreign_cpu, state, \
        (offenders == "" ? "-" : offenders)
    }
  ' <<<"$snapshot") || {
    echo "host contention snapshot was malformed or omitted owner pid $owner_pid" >&2
    return 1
  }
  IFS=$'\t' read -r HOST_CONTENTION_OWNER_PGID \
    HOST_CONTENTION_FOREIGN_CPU_PERCENT HOST_CONTENTION_STATE \
    HOST_CONTENTION_OFFENDERS <<<"$classification"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$sampled_at" \
    "$HOST_CONTENTION_STATE" "$phase" "$HOST_CONTENTION_OWNER_PGID" \
    "$HOST_CONTENTION_FOREIGN_CPU_PERCENT" "$HOST_CONTENTION_OFFENDERS" \
    >>"$log_file"
}

host_contention_require_quiet() {
  local phase=$1

  [[ "$HOST_CONTENTION_STATE" == quiet ]] || {
    echo "calibrated phase $phase observed foreign_cpu_pct=${HOST_CONTENTION_FOREIGN_CPU_PERCENT:-<unknown>} offenders=${HOST_CONTENTION_OFFENDERS:-<unknown>}" >&2
    return 1
  }
}

thermal_validate_state() {
  local state=$1

  case "$state" in
    nominal|fair|serious|critical) ;;
    *)
      echo "malformed macOS thermal state: ${state:-<empty>}" >&2
      return 1
      ;;
  esac
}

thermal_prepare_probe() {
  local swiftc_bin=${HF2Q_THERMAL_SWIFTC_BIN:-/usr/bin/swiftc}
  local probe_source=${HF2Q_THERMAL_PROBE_SOURCE:-$_HF2Q_THERMAL_GUARD_DIR/macos_thermal_probe.swift}
  local probe_dir
  local probe_bin
  local compiler_version
  local compile_error
  local state

  if [[ -n "$THERMAL_PROBE_BIN" ]]; then
    [[ -x "$THERMAL_PROBE_BIN" ]] || {
      echo "prepared thermal-state probe is not executable: $THERMAL_PROBE_BIN" >&2
      return 1
    }
    return 0
  fi
  if [[ -n ${HF2Q_THERMAL_PROBE_BIN:-} ]]; then
    [[ -x "$HF2Q_THERMAL_PROBE_BIN" ]] || {
      echo "thermal-state probe is not executable: $HF2Q_THERMAL_PROBE_BIN" >&2
      return 1
    }
    probe_bin=$HF2Q_THERMAL_PROBE_BIN
    probe_source=""
    swiftc_bin=""
  else
    [[ -x "$swiftc_bin" ]] || {
      echo "thermal-state compiler is not executable: $swiftc_bin" >&2
      return 1
    }
    [[ -f "$probe_source" ]] || {
      echo "thermal-state probe source is missing: $probe_source" >&2
      return 1
    }
    compiler_version=$("$swiftc_bin" --version 2>&1) || {
      echo "failed to identify macOS thermal-state compiler" >&2
      return 1
    }
    [[ -n "$compiler_version" ]] || {
      echo "macOS thermal-state compiler returned an empty version" >&2
      return 1
    }
    probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-thermal-probe.XXXXXX") || {
      echo "failed to create private thermal-state probe directory" >&2
      return 1
    }
    probe_bin="$probe_dir/macos-thermal-probe"
    if ! compile_error=$("$swiftc_bin" -O -whole-module-optimization \
      -o "$probe_bin" "$probe_source" 2>&1); then
      rm -f -- "$probe_bin"
      rmdir -- "$probe_dir" 2>/dev/null || true
      echo "failed to compile macOS thermal-state probe" >&2
      [[ -z "$compile_error" ]] || printf '%s\n' "$compile_error" >&2
      return 1
    fi
    [[ -x "$probe_bin" ]] || {
      rm -f -- "$probe_bin"
      rmdir -- "$probe_dir" 2>/dev/null || true
      echo "compiled thermal-state probe is not executable: $probe_bin" >&2
      return 1
    }
  fi
  state=$("$probe_bin" 2>/dev/null) || {
    if [[ -n ${probe_dir:-} ]]; then
      rm -f -- "$probe_bin"
      rmdir -- "$probe_dir" 2>/dev/null || true
    fi
    echo "failed to read macOS thermal state" >&2
    return 1
  }
  if ! thermal_validate_state "$state"; then
    if [[ -n ${probe_dir:-} ]]; then
      rm -f -- "$probe_bin"
      rmdir -- "$probe_dir" 2>/dev/null || true
    fi
    return 1
  fi
  THERMAL_PROBE_BIN=$probe_bin
  THERMAL_PROBE_OWNED_DIR=${probe_dir:-}
  THERMAL_PROBE_SOURCE=${probe_source:-}
  THERMAL_PROBE_COMPILER=${swiftc_bin:-}
  THERMAL_PROBE_COMPILER_VERSION=${compiler_version:-}
}

thermal_cleanup_probe() {
  if [[ -n "$THERMAL_PROBE_OWNED_DIR" ]]; then
    [[ "$THERMAL_PROBE_BIN" == "$THERMAL_PROBE_OWNED_DIR/macos-thermal-probe" ]] || {
      echo "refusing to clean mismatched thermal-state probe path" >&2
      return 1
    }
    rm -f -- "$THERMAL_PROBE_BIN" || return 1
    rmdir -- "$THERMAL_PROBE_OWNED_DIR" || return 1
  fi
  THERMAL_PROBE_BIN=""
  THERMAL_PROBE_OWNED_DIR=""
  # These exported-by-source globals are consumed by receipt producers after
  # prepare and deliberately cleared here; this helper cannot see those reads.
  # shellcheck disable=SC2034
  THERMAL_PROBE_SOURCE=""
  # shellcheck disable=SC2034
  THERMAL_PROBE_COMPILER=""
  # shellcheck disable=SC2034
  THERMAL_PROBE_COMPILER_VERSION=""
}

thermal_read_state() {
  local state

  thermal_prepare_probe || return 1
  state=$("$THERMAL_PROBE_BIN" 2>/dev/null) || {
    echo "failed to read macOS thermal state" >&2
    return 1
  }
  thermal_validate_state "$state" || return 1
  THERMAL_STATE=$state
}

thermal_sample() {
  local log_file=$1
  local phase=$2
  local sampled_at

  thermal_read_state || return 1
  sampled_at=$(date +%s) || return 1
  [[ "$sampled_at" =~ ^[0-9]+$ ]] || {
    echo "malformed thermal sample timestamp: $sampled_at" >&2
    return 1
  }
  printf '%s\t%s\t%s\n' "$sampled_at" "$THERMAL_STATE" "$phase" \
    >>"$log_file"
  THERMAL_SAMPLED_AT=$sampled_at
}

thermal_wait_for_nominal() {
  local log_file=$1
  local phase=$2
  local settle_seconds=$3
  local timeout_seconds=$4
  local sample_seconds=$5
  local contention_log=${6:-}
  local contention_owner_pid=${7:-}
  local contention_owned_server_pid=${8:-}
  local deadline
  local nominal_since

  for value in "$settle_seconds" "$timeout_seconds" "$sample_seconds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "thermal timing values must be non-negative integers" >&2
      return 2
    }
  done
  deadline=$((SECONDS + timeout_seconds))
  nominal_since=-1
  : >"$log_file"
  if [[ -n "$contention_log" || -n "$contention_owner_pid" ]]; then
    [[ -n "$contention_log" && -n "$contention_owner_pid" ]] || {
      echo "thermal settle requires both contention log and owner pid" >&2
      return 2
    }
    : >"$contention_log"
  fi
  while :; do
    thermal_sample "$log_file" "$phase" || return 1
    if [[ -n "$contention_log" ]]; then
      host_contention_sample "$contention_log" "$phase" \
        "$contention_owner_pid" "$THERMAL_SAMPLED_AT" \
        "$contention_owned_server_pid" || return 1
    else
      HOST_CONTENTION_STATE=quiet
    fi
    if [[ "$THERMAL_STATE" == nominal \
      && "$HOST_CONTENTION_STATE" == quiet ]]; then
      if ((nominal_since < 0)); then
        nominal_since=$SECONDS
      fi
      if ((SECONDS - nominal_since >= settle_seconds)); then
        return 0
      fi
    else
      nominal_since=-1
    fi
    if ((SECONDS >= deadline)); then
      echo "thermal state and host work did not remain calibrated for ${settle_seconds}s within ${timeout_seconds}s" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

thermal_monitor_nominal() {
  local log_file=$1
  local phase=$2
  local stop_file=$3
  local sample_seconds=$4
  local contention_log=${5:-}
  local contention_owner_pid=${6:-}

  [[ "$sample_seconds" =~ ^[0-9]+$ ]] || {
    echo "thermal sample interval must be a non-negative integer" >&2
    return 2
  }
  while [[ ! -e "$stop_file" ]]; do
    thermal_sample "$log_file" "$phase" || return 1
    if [[ -n "$contention_log" ]]; then
      [[ -n "$contention_owner_pid" ]] || return 2
      host_contention_sample "$contention_log" "$phase" \
        "$contention_owner_pid" "$THERMAL_SAMPLED_AT" || return 1
      host_contention_require_quiet "$phase" || return 1
    fi
    if [[ "$THERMAL_STATE" != nominal ]]; then
      echo "calibrated phase $phase observed non-nominal thermal state: $THERMAL_STATE" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

thermal_read_process_state() {
  local producer_pid=$1
  local state

  if ! state=$(/bin/ps -p "$producer_pid" -o state= 2>/dev/null \
    | tr -d '[:space:]'); then
    if kill -0 "$producer_pid" 2>/dev/null; then
      echo "failed to read live thermal producer state: $producer_pid" >&2
      return 1
    fi
    THERMAL_PROCESS_STATE=""
    return 0
  fi
  if [[ -z "$state" ]] && kill -0 "$producer_pid" 2>/dev/null; then
    echo "thermal producer state was empty for live pid: $producer_pid" >&2
    return 1
  fi
  THERMAL_PROCESS_STATE=$state
}

thermal_monitor_nominal_while_pid() {
  local log_file=$1
  local phase=$2
  local producer_pid=$3
  local sample_seconds=$4
  local contention_log=${5:-}
  local contention_owner_pid=${6:-}

  [[ "$producer_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "thermal producer pid must be a positive integer" >&2
    return 2
  }
  [[ "$sample_seconds" =~ ^[0-9]+$ ]] || {
    echo "thermal sample interval must be a non-negative integer" >&2
    return 2
  }
  while :; do
    thermal_read_process_state "$producer_pid" || return 1
    if [[ -z "$THERMAL_PROCESS_STATE" || "$THERMAL_PROCESS_STATE" == Z* ]]; then
      return 0
    fi
    thermal_sample "$log_file" "$phase" || return 1
    if [[ -n "$contention_log" ]]; then
      [[ -n "$contention_owner_pid" ]] || return 2
      host_contention_sample "$contention_log" "$phase" \
        "$contention_owner_pid" "$THERMAL_SAMPLED_AT" || return 1
      host_contention_require_quiet "$phase" || return 1
    fi
    if [[ "$THERMAL_STATE" != nominal ]]; then
      echo "calibrated phase $phase observed non-nominal thermal state: $THERMAL_STATE" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

thermal_monitor_fair_or_better_while_pid() {
  local log_file=$1
  local phase=$2
  local producer_pid=$3
  local sample_seconds=$4
  local contention_log=${5:-}
  local contention_owner_pid=${6:-}
  local contention_owned_server_pid=${7:-}

  [[ "$producer_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "thermal producer pid must be a positive integer" >&2
    return 2
  }
  [[ "$sample_seconds" =~ ^[0-9]+$ ]] || {
    echo "thermal sample interval must be a non-negative integer" >&2
    return 2
  }
  while :; do
    thermal_read_process_state "$producer_pid" || return 1
    if [[ -z "$THERMAL_PROCESS_STATE" || "$THERMAL_PROCESS_STATE" == Z* ]]; then
      return 0
    fi
    thermal_sample "$log_file" "$phase" || return 1
    if [[ -n "$contention_log" ]]; then
      [[ -n "$contention_owner_pid" ]] || return 2
      host_contention_sample "$contention_log" "$phase" \
        "$contention_owner_pid" "$THERMAL_SAMPLED_AT" \
        "$contention_owned_server_pid" || return 1
      host_contention_require_quiet "$phase" || return 1
    fi
    case "$THERMAL_STATE" in
      nominal|fair) ;;
      serious|critical)
        echo "calibrated phase $phase exceeded fair thermal state: $THERMAL_STATE" >&2
        return 1
        ;;
      *)
        echo "calibrated phase $phase observed invalid thermal state: $THERMAL_STATE" >&2
        return 1
        ;;
    esac
    sleep "$sample_seconds"
  done
}

thermal_prepare_cold_receipt_dir() {
  local receipt_dir=$1
  local existing_receipt

  mkdir -p "$receipt_dir" || return 1
  existing_receipt=$(find "$receipt_dir" -maxdepth 1 -type f \
    -name 'agent-*.cold.json' -print -quit) || return 1
  if [[ -n "$existing_receipt" ]]; then
    echo "cold-cohort receipt directory contains stale evidence: $existing_receipt" >&2
    return 1
  fi
}

thermal_monitor_nominal_until_cold_receipts() {
  local log_file=$1
  local phase=$2
  local receipt_dir=$3
  local expected_receipts=$4
  local sample_seconds=$5
  local timeout_seconds=$6
  local producer_pid=${7:-}
  local contention_log=${8:-}
  local contention_owner_pid=${9:-}
  local deadline
  local producer_state
  local receipt_index
  local receipt_count
  local receipt_set_complete

  for value in "$expected_receipts" "$sample_seconds" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "cold-cohort thermal values must be non-negative integers" >&2
      return 2
    }
  done
  ((expected_receipts > 0)) || {
    echo "cold-cohort thermal monitor requires at least one receipt" >&2
    return 2
  }
  [[ -d "$receipt_dir" ]] || {
    echo "cold-cohort receipt directory is missing: $receipt_dir" >&2
    return 1
  }

  deadline=$((SECONDS + timeout_seconds))
  while :; do
    if [[ -n "$producer_pid" ]]; then
      producer_state=$(ps -p "$producer_pid" -o state= 2>/dev/null \
        | tr -d '[:space:]')
      if [[ -z "$producer_state" || "$producer_state" == Z* ]]; then
        echo "cold-cohort producer exited before publishing all receipts" >&2
        return 1
      fi
    fi
    thermal_sample "$log_file" "$phase" || return 1
    if [[ -n "$contention_log" ]]; then
      [[ -n "$contention_owner_pid" ]] || return 2
      host_contention_sample "$contention_log" "$phase" \
        "$contention_owner_pid" "$THERMAL_SAMPLED_AT" || return 1
      host_contention_require_quiet "$phase" || return 1
    fi
    if [[ "$THERMAL_STATE" != nominal ]]; then
      echo "calibrated phase $phase observed non-nominal thermal state: $THERMAL_STATE" >&2
      return 1
    fi
    receipt_count=$(find "$receipt_dir" -maxdepth 1 -type f \
      -name 'agent-*.cold.json' -size +0c | wc -l | tr -d '[:space:]')
    [[ "$receipt_count" =~ ^[0-9]+$ ]] || {
      echo "failed to count cold-cohort receipts" >&2
      return 1
    }
    receipt_set_complete=1
    for ((receipt_index = 1; receipt_index <= expected_receipts; receipt_index++)); do
      if [[ ! -s "$receipt_dir/agent-${receipt_index}.cold.json" ]]; then
        receipt_set_complete=0
        break
      fi
    done
    if ((receipt_count == expected_receipts && receipt_set_complete == 0)); then
      echo "cold-cohort receipts do not match agent-1 through agent-${expected_receipts}" >&2
      return 1
    fi
    if ((receipt_count == expected_receipts && receipt_set_complete == 1)); then
      thermal_sample "$log_file" "$phase-end" || return 1
      if [[ -n "$contention_log" ]]; then
        host_contention_sample "$contention_log" "$phase-end" \
          "$contention_owner_pid" "$THERMAL_SAMPLED_AT" || return 1
        host_contention_require_quiet "$phase-end" || return 1
      fi
      [[ "$THERMAL_STATE" == nominal ]] || {
        echo "calibrated phase $phase ended in non-nominal thermal state: $THERMAL_STATE" >&2
        return 1
      }
      return 0
    fi
    if ((receipt_count > expected_receipts)); then
      echo "cold-cohort receipt count exceeded ${expected_receipts}: $receipt_count" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "cold cohort did not publish ${expected_receipts} receipts within ${timeout_seconds}s" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

host_contention_validate_measurement_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local stats

  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" \
    -v cpu_max="$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" '
    BEGIN { invalid = 0; gaps = 0; contended = 0; owner_pgid = "" }
    {
      if (NF != 6 || $1 !~ /^[0-9]+$/ \
          || $2 !~ /^(quiet|contended)$/ || length($3) == 0 \
          || $4 !~ /^[1-9][0-9]*$/ \
          || $5 !~ /^[0-9]+([.][0-9]+)?$/ \
          || cpu_max !~ /^[0-9]+([.][0-9]+)?$/ || cpu_max + 0 <= 0 \
          || ($2 == "quiet" && ($5 + 0 >= cpu_max + 0 || $6 != "-")) \
          || ($2 == "contended" \
            && $5 + 0 < cpu_max + 0 \
            && $6 !~ /^[0-9]+:[0-9]+:[A-Za-z0-9._+-]+(,[0-9]+:[0-9]+:[A-Za-z0-9._+-]+)*$/) \
          || ($6 != "-" \
            && $6 !~ /^[0-9]+:[0-9]+:[A-Za-z0-9._+-]+(,[0-9]+:[0-9]+:[A-Za-z0-9._+-]+)*$/)) {
        invalid++
        next
      }
      samples++
      if (owner_pgid == "") owner_pgid = $4
      if ($4 != owner_pgid) invalid++
      if ($2 == "contended") contended++
      if (samples == 1) first = $1
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      last = $1
    }
    END {
      duration = samples > 0 ? last - first : -1
      printf "%d\t%d\t%d\t%d\t%d\n", samples, duration, \
        contended, gaps, invalid
    }
  ' "$log_file") || return 1
  IFS=$'\t' read -r HOST_CONTENTION_LOG_SAMPLES \
    HOST_CONTENTION_LOG_DURATION_SECONDS HOST_CONTENTION_LOG_CONTENDED_SAMPLES \
    HOST_CONTENTION_LOG_GAPS HOST_CONTENTION_LOG_INVALID_ROWS <<<"$stats"
  ((HOST_CONTENTION_LOG_SAMPLES >= 2)) \
    && ((HOST_CONTENTION_LOG_DURATION_SECONDS > 0)) \
    && ((HOST_CONTENTION_LOG_CONTENDED_SAMPLES == 0)) \
    && ((HOST_CONTENTION_LOG_GAPS == 0)) \
    && ((HOST_CONTENTION_LOG_INVALID_ROWS == 0))
}

host_contention_validate_settle_log() {
  local log_file=$1
  local required_seconds=$2
  local maximum_gap_seconds=$3
  local stats

  [[ "$required_seconds" =~ ^[0-9]+$ ]] || return 2
  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" \
    -v cpu_max="$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" '
    BEGIN {
      invalid = 0; gaps = 0; contended = 0; quiet_since = -1
      owner_pgid = ""
    }
    {
      if (NF != 6 || $1 !~ /^[0-9]+$/ \
          || $2 !~ /^(quiet|contended)$/ || length($3) == 0 \
          || $4 !~ /^[1-9][0-9]*$/ \
          || $5 !~ /^[0-9]+([.][0-9]+)?$/ \
          || cpu_max !~ /^[0-9]+([.][0-9]+)?$/ || cpu_max + 0 <= 0 \
          || ($2 == "quiet" && ($5 + 0 >= cpu_max + 0 || $6 != "-")) \
          || ($2 == "contended" \
            && $5 + 0 < cpu_max + 0 \
            && $6 !~ /^[0-9]+:[0-9]+:[A-Za-z0-9._+-]+(,[0-9]+:[0-9]+:[A-Za-z0-9._+-]+)*$/) \
          || ($6 != "-" \
            && $6 !~ /^[0-9]+:[0-9]+:[A-Za-z0-9._+-]+(,[0-9]+:[0-9]+:[A-Za-z0-9._+-]+)*$/)) {
        invalid++
        next
      }
      samples++
      if (owner_pgid == "") owner_pgid = $4
      if ($4 != owner_pgid) invalid++
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      if ($2 == "contended") {
        contended++
        quiet_since = -1
      } else if (quiet_since < 0) {
        quiet_since = $1
      }
      last = $1
    }
    END {
      duration = quiet_since >= 0 ? last - quiet_since : -1
      printf "%d\t%d\t%d\t%d\t%d\n", samples, duration, \
        contended, gaps, invalid
    }
  ' "$log_file") || return 1
  IFS=$'\t' read -r HOST_CONTENTION_LOG_SAMPLES \
    HOST_CONTENTION_LOG_DURATION_SECONDS HOST_CONTENTION_LOG_CONTENDED_SAMPLES \
    HOST_CONTENTION_LOG_GAPS HOST_CONTENTION_LOG_INVALID_ROWS <<<"$stats"
  ((HOST_CONTENTION_LOG_SAMPLES > 0)) \
    && ((HOST_CONTENTION_LOG_DURATION_SECONDS >= required_seconds)) \
    && ((HOST_CONTENTION_LOG_GAPS == 0)) \
    && ((HOST_CONTENTION_LOG_INVALID_ROWS == 0))
}

host_contention_validate_thermal_alignment() {
  local thermal_log=$1
  local contention_log=$2

  cmp -s \
    <(awk -F '\t' 'NF == 3 { print $1 "\t" $3 }' "$thermal_log") \
    <(awk -F '\t' 'NF == 6 { print $1 "\t" $3 }' "$contention_log")
}

thermal_validate_measurement_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local stats

  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" '
    BEGIN { invalid = 0; gaps = 0; non_nominal = 0 }
    {
      if (NF != 3 || $1 !~ /^[0-9]+$/) {
        invalid++
        next
      }
      samples++
      if ($2 != "nominal") non_nominal++
      if (samples == 1) first = $1
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      last = $1
    }
    END {
      duration = samples > 0 ? last - first : -1
      printf "%d\t%d\t%d\t%d\t%d\n", samples, duration,
        non_nominal, gaps, invalid
    }
  ' "$log_file") || return 1
  # These globals are the validator's output contract for release callers.
  # shellcheck disable=SC2034
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_GAPS THERMAL_LOG_INVALID_ROWS \
    <<<"$stats"
  ((THERMAL_LOG_SAMPLES >= 2)) \
    && ((THERMAL_LOG_DURATION_SECONDS > 0)) \
    && ((THERMAL_LOG_NON_NOMINAL_SAMPLES == 0)) \
    && ((THERMAL_LOG_GAPS == 0)) \
    && ((THERMAL_LOG_INVALID_ROWS == 0))
}

thermal_validate_fair_or_better_measurement_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local stats

  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" '
    BEGIN { invalid = 0; gaps = 0; non_nominal = 0; fair = 0; over_limit = 0 }
    {
      if (NF != 3 || $1 !~ /^[0-9]+$/ \
          || $2 !~ /^(nominal|fair|serious|critical)$/) {
        invalid++
        next
      }
      samples++
      if ($2 != "nominal") non_nominal++
      if ($2 == "fair") fair++
      if ($2 == "serious" || $2 == "critical") over_limit++
      if (samples == 1) first = $1
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      last = $1
    }
    END {
      duration = samples > 0 ? last - first : -1
      printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", samples, duration,
        non_nominal, fair, over_limit, gaps, invalid
    }
  ' "$log_file") || return 1
  # These globals are the validator's output contract for release callers.
  # shellcheck disable=SC2034
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_FAIR_SAMPLES \
    THERMAL_LOG_OVER_LIMIT_SAMPLES THERMAL_LOG_GAPS \
    THERMAL_LOG_INVALID_ROWS <<<"$stats"
  ((THERMAL_LOG_SAMPLES >= 2)) \
    && ((THERMAL_LOG_DURATION_SECONDS > 0)) \
    && ((THERMAL_LOG_OVER_LIMIT_SAMPLES == 0)) \
    && ((THERMAL_LOG_GAPS == 0)) \
    && ((THERMAL_LOG_INVALID_ROWS == 0))
}

thermal_validate_settle_log() {
  local log_file=$1
  local required_seconds=$2
  local maximum_gap_seconds=$3
  local stats

  [[ "$required_seconds" =~ ^[0-9]+$ ]] || return 2
  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" '
    BEGIN { invalid = 0; gaps = 0; nominal_since = -1 }
    {
      if (NF != 3 || $1 !~ /^[0-9]+$/ \
          || $2 !~ /^(nominal|fair|serious|critical)$/) {
        invalid++
        next
      }
      samples++
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      if ($2 != "nominal") {
        non_nominal++
        nominal_since = -1
      } else if (nominal_since < 0) {
        nominal_since = $1
      }
      last = $1
    }
    END {
      duration = nominal_since >= 0 ? last - nominal_since : -1
      printf "%d\t%d\t%d\t%d\t%d\n", samples, duration,
        non_nominal, gaps, invalid
    }
  ' "$log_file") || return 1
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_GAPS THERMAL_LOG_INVALID_ROWS \
    <<<"$stats"
  ((THERMAL_LOG_SAMPLES > 0)) \
    && ((THERMAL_LOG_DURATION_SECONDS >= required_seconds)) \
    && ((THERMAL_LOG_GAPS == 0)) \
    && ((THERMAL_LOG_INVALID_ROWS == 0))
}
