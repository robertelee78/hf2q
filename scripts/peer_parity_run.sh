#!/usr/bin/env bash
# scripts/peer_parity_run.sh — ADR-014 P10 cold-cache benchmark wrapper.
#
# Per ADR-014 Decision 15 (lines 575-602):
#   1 warmup run (discarded) → 60s thermal cooldown → 3 timed runs
#
# Robert lock 2026-04-25 round 2: the Apple-Silicon ML benchmark
# community's actual practice is `warmup → cooldown → timed runs`, not
# "system reboot, single cold run." Sources cited inline in ADR-014.
# Warmup-discarded ensures both peers measure with warm Metal-shader
# caches; the 60s cooldown holds the M5 Max within the same thermal
# envelope across the three timed iterations.
#
# Usage:
#   peer_parity_run.sh <peer> <model> <variant> <output_csv>
#
# Where:
#   <peer>       — one of: llama_quantize | llama_imatrix |
#                  convert_hf_to_gguf | mlx_lm_convert
#   <model>      — path to the input model directory (HF format) or
#                  GGUF file (depending on <peer>)
#   <variant>    — quant variant string (e.g. Q4_K_M, dwq-4-6)
#   <output_csv> — path to the CSV file to populate; the script writes
#                  a header + 3 rows sorted ascending by wall_s so the
#                  median is row 2.
#
# Output CSV schema (Decision 15 surface):
#   run_idx,wall_s,peak_rss_bytes,exit_code
#
# This script is validated with `bash -n` in CI and is NOT executed
# against real peers in P10 iter-1 (real-model runs land in P11).

set -euo pipefail

# ---------------------------------------------------------------------
# 1. Argument validation
# ---------------------------------------------------------------------

if [ "$#" -ne 4 ]; then
    cat >&2 <<USAGE
ERROR: expected 4 positional args; got $#.

usage: $0 <peer> <model> <variant> <output_csv>

  <peer>       — one of: llama_quantize | llama_imatrix |
                 convert_hf_to_gguf | mlx_lm_convert
  <model>      — path to the input model
  <variant>    — quant variant string (e.g. Q4_K_M, dwq-4-6)
  <output_csv> — path to the CSV file to write
USAGE
    exit 2
fi

PEER=$1
MODEL=$2
VARIANT=$3
OUT=$4

# ---------------------------------------------------------------------
# 2. Resolve the per-peer command
# ---------------------------------------------------------------------
#
# Each branch builds the `CMD` array which is later spliced under
# `/usr/bin/time -l` (BSD/Mac format, NOT GNU `-v`). Operator overrides
# match the Rust harness's `HF2Q_*_BIN` env-var contract so a single
# pinned build of each peer drives both Cargo tests and this shell
# wrapper.

case "$PEER" in
    llama_quantize)
        BIN=${HF2Q_LLAMA_QUANTIZE_BIN:-llama-quantize}
        CMD=("$BIN" "$MODEL" "${MODEL%.*}-${VARIANT}.gguf" "$VARIANT")
        ;;
    llama_imatrix)
        BIN=${HF2Q_LLAMA_IMATRIX_BIN:-llama-imatrix}
        CMD=("$BIN" -m "$MODEL" -f "${HF2Q_IMATRIX_CALIBRATION_TEXT:-/dev/null}" -o "${MODEL%.*}-${VARIANT}.imatrix.dat")
        ;;
    convert_hf_to_gguf)
        BIN=${HF2Q_LLAMA_CONVERT_HF_BIN:-convert_hf_to_gguf.py}
        CMD=("$BIN" "$MODEL" --outfile "${MODEL%/}-${VARIANT}.gguf")
        ;;
    mlx_lm_convert)
        PY=${HF2Q_PYTHON_BIN:-python3}
        CMD=("$PY" -m mlx_lm.convert --hf-path "$MODEL" --mlx-path "${MODEL%/}-mlx-${VARIANT}" --quantize)
        ;;
    *)
        echo "ERROR: unknown peer '$PEER'" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------
# 3. Cold-cache protocol: 1 warmup discarded
# ---------------------------------------------------------------------
#
# The warmup run primes Metal-shader / KV-cache state so subsequent
# timed runs measure steady-state. We swallow the warmup's output (and
# allow it to fail without aborting — set +e around the call) because
# the peer might legitimately fail on warmup if e.g. the output file
# already exists from a prior crashed run.

echo "[peer_parity_run] warmup discarded (1 run) — peer=$PEER variant=$VARIANT" >&2
set +e
/usr/bin/time -l "${CMD[@]}" >/dev/null 2>/dev/null
set -e

# ---------------------------------------------------------------------
# 4. 60s thermal cooldown
# ---------------------------------------------------------------------
#
# Per ADR-014 Decision 15: holds the M5 Max within the same thermal
# envelope across the three timed iterations. Sourced from
# mlx_transformers_benchmark's cooldown_time_fraction practice.

echo "[peer_parity_run] 60s thermal cooldown" >&2
sleep 60

# ---------------------------------------------------------------------
# 5. 3 timed runs
# ---------------------------------------------------------------------
#
# Each run is wrapped in `/usr/bin/time -l` (BSD/Mac format). We parse:
#   - "real" wall-clock from the first time line (e.g. "1.23 real")
#   - "maximum resident set size" peak RSS in bytes
# and append a CSV row per run. Final step sorts the CSV body by wall_s
# ascending so the harness reads the median (row 2 of 3).

echo "run_idx,wall_s,peak_rss_bytes,exit_code" > "$OUT"

for i in 1 2 3; do
    TIME_FILE="$OUT.run$i.time"
    set +e
    /usr/bin/time -l "${CMD[@]}" 1>/dev/null 2> "$TIME_FILE"
    EXIT=$?
    set -e

    # Parse `real` wall-clock from the BSD time stderr block. macOS
    # emits `        1.23 real         0.45 user         0.06 sys` as
    # the first non-empty line.
    WALL=$(awk '
        /real/ {
            for (j = 1; j <= NF; j++) {
                if ($j == "real") {
                    print $(j-1);
                    exit;
                }
            }
        }
    ' "$TIME_FILE")
    if [ -z "$WALL" ]; then
        WALL=-1.0
    fi

    # Parse `maximum resident set size` (bytes on macOS).
    RSS=$(awk '
        /maximum resident set size/ {
            print $1;
            exit;
        }
    ' "$TIME_FILE")
    if [ -z "$RSS" ]; then
        # u64::MAX in decimal — same sentinel the Rust harness uses.
        RSS=18446744073709551615
    fi

    echo "$i,$WALL,$RSS,$EXIT" >> "$OUT"
done

# ---------------------------------------------------------------------
# 6. Sort CSV body by wall_s ascending so median = row 2 of 3
# ---------------------------------------------------------------------

HEADER=$(head -n 1 "$OUT")
SORTED=$(tail -n +2 "$OUT" | sort -t, -k2,2 -g)
{
    echo "$HEADER"
    echo "$SORTED"
} > "$OUT"

echo "[peer_parity_run] wrote $OUT (3 runs sorted ascending by wall_s)" >&2
