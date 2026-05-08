#!/usr/bin/env bash
# ADR-020 §8.3 AC #6 + #7 + #8 — per-family E2E validation harness.
#
# Runs `hf2q dwq-train --bench` on the canonical 4 family × bit-pair
# combos called out by §8.3 AC #8 and parses the per-run output for the
# §8.3 AC #7 acceptance gate.  Per-family PASS/FAIL summary at the end.
#
# Combos (AC #8):
#   1. Qwen 3.6 35B-A3B-Abliterix-EGA × dwq-4
#   2. Qwen 3.6 35B-A3B-Abliterix-EGA × dwq-6
#   3. Gemma 4 26B-A4B-it-ara         × dwq-4
#   4. Gemma 4 26B-A4B-it-ara         × dwq-6
#
# Acceptance gates:
#   AC #6  — peak RSS < 100 GB.  --rss-cap-gb 100 hard-aborts otherwise.
#   AC #7  — mean_delta_kl_nats > 0.05 over the trained Linears.
#   AC #8  — every combo PASSes both AC #6 and AC #7.
#
# Usage:
#   scripts/adr020_ac8_validate.sh [--limit N] [--smoke]
#
# Flags:
#   --limit N : Cap each combo at N Linears (default: full run).  Useful
#               for first-time validation without committing hours of GPU.
#   --smoke   : Shortcut for `--limit 5` — minimal-cost wiring check.
#
# Per-combo output:  /tmp/adr020_ac8/<family>_dwq-<bits>.{out,err,safetensors}
# Aggregate report:  /tmp/adr020_ac8/SUMMARY.txt
#
# Exit code:
#   0  — all 4 combos PASS AC #6 + AC #7 (AC #8 closed)
#   1  — at least one combo FAILed any gate
#   2  — script-internal error (missing GGUF, build failure, …)

set -u
set -o pipefail

# ─────────────── arg parsing ───────────────
LIMIT_ARG=()
SMOKE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            LIMIT_ARG=(--limit "$2")
            shift 2
            ;;
        --smoke)
            SMOKE=1
            LIMIT_ARG=(--limit 5)
            shift
            ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: $1" >&2
            exit 2
            ;;
    esac
done

# ─────────────── repo + binary ───────────────
REPO="/opt/hf2q"
cd "$REPO" || { echo "cd $REPO failed" >&2; exit 2; }

OUT_DIR="/tmp/adr020_ac8"
mkdir -p "$OUT_DIR"

echo "[ac8] building hf2q (release) ..."
cargo build --release --bin hf2q 2>&1 | tail -3
HF2Q="$REPO/target/release/hf2q"
[[ -x "$HF2Q" ]] || { echo "hf2q binary missing at $HF2Q" >&2; exit 2; }

# ─────────────── canonical GGUFs ───────────────
QWEN_GGUF="$REPO/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf"
GEMMA_GGUF="$REPO/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf"

for f in "$QWEN_GGUF" "$GEMMA_GGUF"; do
    [[ -f "$f" ]] || { echo "missing GGUF: $f" >&2; exit 2; }
done

# ─────────────── per-combo runner ───────────────
COMBOS=(
    "qwen35moe-apex|$QWEN_GGUF|4"
    "qwen35moe-apex|$QWEN_GGUF|6"
    "gemma4-26b-ara|$GEMMA_GGUF|4"
    "gemma4-26b-ara|$GEMMA_GGUF|6"
)

PASS_COUNT=0
FAIL_COUNT=0
SUMMARY="$OUT_DIR/SUMMARY.txt"
: > "$SUMMARY"

for combo in "${COMBOS[@]}"; do
    family="${combo%%|*}"
    rest="${combo#*|}"
    gguf="${rest%|*}"
    bits="${rest##*|}"
    label="${family}_dwq-${bits}"
    out_path="$OUT_DIR/${label}.safetensors"
    log_out="$OUT_DIR/${label}.out"
    log_err="$OUT_DIR/${label}.err"

    echo "============================================================"
    echo "[ac8] running combo: $label"
    echo "[ac8]   gguf: $gguf"
    echo "[ac8]   bits: $bits"
    echo "[ac8]   limit: ${LIMIT_ARG[*]:-(none)}"
    echo "============================================================"

    # ADR-020 iter-12g: perturb_factor=1.0 is the production-correct
    # default (apples-to-apples vs Q4_0 baseline).
    "$HF2Q" dwq-train \
        --gguf "$gguf" \
        --output "$out_path" \
        --bits "$bits" \
        --bench \
        --rss-cap-gb 100 \
        --perturb-factor 1.0 \
        "${LIMIT_ARG[@]}" \
        > "$log_out" 2> "$log_err"
    rc=$?

    # Parse the bench summary line.  Format from src/main.rs:323-327:
    #   [dwq-train] bench summary: mean_delta_kl_nats = ±0.NNNNNN
    #   across N Linears (§8.3 AC #7 threshold +0.05: PASS|FAIL)
    bench_line=$(grep -m1 "bench summary" "$log_out" 2>/dev/null || true)

    # AC #6: rss-cap-gb hard-aborts on exceed (Err propagates to exit 1).
    # AC #7: gate is read from the bench line.
    ac6=""
    ac7=""
    if [[ $rc -eq 0 && -n "$bench_line" ]]; then
        ac6="PASS"
        if echo "$bench_line" | grep -q "PASS)$"; then
            ac7="PASS"
        elif echo "$bench_line" | grep -q "FAIL)$"; then
            ac7="FAIL"
        else
            ac7="?? (parse failure on bench line)"
        fi
    elif [[ $rc -ne 0 ]]; then
        ac6="FAIL (exit $rc — see $log_err)"
        ac7="N/A (training aborted)"
    else
        ac6="PASS"
        ac7="?? (no bench summary in stdout)"
    fi

    combo_pass=0
    if [[ "$ac6" == "PASS" && "$ac7" == "PASS" ]]; then
        combo_pass=1
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    {
        printf "%-25s  bits=%d  AC#6: %-8s  AC#7: %s\n" \
            "$family" "$bits" "$ac6" "$ac7"
        if [[ -n "$bench_line" ]]; then
            echo "    $bench_line"
        fi
        if [[ $combo_pass -eq 0 ]]; then
            echo "    last 3 lines of $log_err:"
            tail -3 "$log_err" 2>/dev/null | sed 's/^/      /'
        fi
    } | tee -a "$SUMMARY"
done

echo "============================================================"
echo "[ac8] §8.3 AC #8 per-family pass: $PASS_COUNT / 4"
echo "[ac8] summary written to $SUMMARY"
echo "============================================================"
cat "$SUMMARY"

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "[ac8] PASS — AC #8 closed (all 4 combos meet AC #6 + AC #7)"
    exit 0
else
    echo "[ac8] FAIL — $FAIL_COUNT of 4 combos failed at least one gate"
    exit 1
fi
