#!/usr/bin/env bash
# scripts/profile-p3aprime.sh
#
# ADR-015 Wave 2a P3a' — live profile pass for the qwen35 decode hot path.
#
# Tooling decision (queen-prescribed, accepted by perf-engineer 2026-04-26):
#   Use `xctrace record --template "Time Profiler"` (Instruments TimeProfiler)
#   from the system Xcode toolchain at:
#     /Applications/Xcode.app/Contents/Developer/usr/bin/xctrace
#
#   Rationale (vs `cargo flamegraph` / DTrace):
#     - xctrace is native-macOS, post-SIP-stable, captures Rust release-mode
#       frames + Metal/IOKit driver frames in one trace, and resolves Rust
#       symbols against the binary's DWARF without a special build.
#     - cargo-flamegraph drives DTrace, which on macOS 26 / Apple Silicon
#       requires SIP modifications. We avoid SIP modifications entirely.
#     - Apple's Instruments-based GPU counters (Q-NAX-1 / Q-NAX-4 in ADR-015)
#       live in the same trace family — same tool family for future P3c work.
#
#   Fallback: if xctrace symbolication of Rust release-mode frames is poor
#   (e.g. only `_$LT$...$GT$` mangled names visible), install
#   `cargo install flamegraph` and switch to:
#     CARGO_PROFILE_RELEASE_DEBUG=true flamegraph -o flame.svg --root -- <cmd>
#   That fallback is documented but not used by default.
#
# Fixture decision (queen-prescribed, accepted by perf-engineer 2026-04-26;
# corrected per Codex review AF2, applied 2026-04-27):
#   /opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
#     - qwen35 dense (Qwen3.5-derivative)
#     - exercises forward_gpu.rs + gpu_full_attn.rs + the dense-quantized
#       FFN path (`build_dense_ffn_layer_gpu_q` at gpu_ffn.rs:578), which
#       calls `quantized_matmul_ggml` directly — it does NOT call the
#       `fn proj` helper at gpu_ffn.rs:347-423. Earlier iterations of this
#       comment incorrectly claimed the dense fixture exercises the
#       `gpu_ffn proj path`; ADR-015 §P3a' Q4 (Codex review) corrects this.
#     - fits comfortably in 128 GB unified, no OOM
#     - shares the encoder + alloc_buffer + apply_imrope + DeltaNet hot
#       frames with the apex 35B-A3B MoE; does NOT share the MoE expert
#       routing path (`build_moe_ffn_layer_gpu_q`) or the literal
#       unquantized `fn proj` allocation.
#   Coverage gap (documented in ADR-015 §P3a'):
#     - The literal `fn proj` site (gpu_ffn.rs:397-404) is NOT exercised
#       by this dense fixture. H1's literal-site verdict is therefore
#       LITERAL SITE NOT MEASURED on this fixture; the dense-FFN-Q alloc
#       category analog (5-6 intermediate alloc_buffer calls per FFN
#       layer at gpu_ffn.rs:592-606) IS exercised and is the dense
#       analog measured here.
#     - MoE literal H1 + apex-MoE 288 µs residual decomposition remain
#       Wave 2b. Wave 2b must run the apex 35B-A3B MoE fixture before
#       any P3b reduction lands at the literal `fn proj` site.
#
# Cold-SoC methodology (per feedback_perf_gate_thermal_methodology):
#   - 60-second sleep between trials (queen recommendation).
#   - `pmset -g therm` snapshot captured pre-trial → if CPU_Speed_Limit < 100,
#     the trial is rejected and the script aborts with status 2.
#   - 3 trials minimum.
#
# Outputs:
#   /tmp/cfa-adr015-wave2a-p3a-prime/trace-<trial>-<date>.trace
#   /tmp/cfa-adr015-wave2a-p3a-prime/trial-<trial>-<date>.metadata.json
#   /tmp/cfa-adr015-wave2a-p3a-prime/trial-<trial>-<date>.stdout
#   /tmp/cfa-adr015-wave2a-p3a-prime/trial-<trial>-<date>.stderr
#   /tmp/cfa-adr015-wave2a-p3a-prime/topcalls-<trial>-<date>.txt   (xctrace export)
#   docs/perf-traces/.gitkeep                                       (committed)
#
# Trace artifacts (.trace, .stdout, .stderr) live under /tmp ONLY — they are
# large (>100 MB), not deterministic, and are referenced by path from
# ADR-015 §P3a' rather than committed to git.
#
# Usage:
#   scripts/profile-p3aprime.sh                # 3 trials, 64 decode tokens
#   N_TOKENS=128 N_TRIALS=5 scripts/profile-p3aprime.sh
#   SKIP_THERMAL_GATE=1 scripts/profile-p3aprime.sh   # CI / quick smoke only

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
FIXTURE="${FIXTURE:-/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf}"
PROMPT="${PROMPT:-Hello, my name is}"
N_TOKENS="${N_TOKENS:-64}"
N_TRIALS="${N_TRIALS:-3}"
THERMAL_SETTLE_SEC="${THERMAL_SETTLE_SEC:-60}"
OUT_DIR="${OUT_DIR:-/tmp/cfa-adr015-wave2a-p3a-prime}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
XCTRACE="${XCTRACE:-/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace}"

mkdir -p "$OUT_DIR"

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------
if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "ERROR: hf2q binary not found at $HF2Q_BIN" >&2
  echo "Build with: cargo build --release -p hf2q" >&2
  exit 1
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "ERROR: fixture not found at $FIXTURE" >&2
  exit 1
fi
if [[ ! -x "$XCTRACE" ]]; then
  echo "ERROR: xctrace not found at $XCTRACE" >&2
  echo "Install Xcode command-line tools, or override XCTRACE=." >&2
  exit 1
fi

echo "=== ADR-015 Wave 2a P3a' live profile pass ==="
echo "binary    : $HF2Q_BIN"
echo "fixture   : $FIXTURE"
echo "prompt    : $PROMPT"
echo "n_tokens  : $N_TOKENS"
echo "n_trials  : $N_TRIALS"
echo "settle    : ${THERMAL_SETTLE_SEC}s between trials"
echo "out_dir   : $OUT_DIR"
echo

# Capture run-wide hardware metadata once.
RUN_META="$OUT_DIR/run-${DATE_TAG}.metadata.json"
{
  echo "{"
  echo "  \"date_utc\": \"$DATE_TAG\","
  echo "  \"hostname\": \"$(hostname -s)\","
  echo "  \"macos_version\": \"$(sw_vers -productVersion)\","
  echo "  \"macos_build\": \"$(sw_vers -buildVersion)\","
  echo "  \"chip\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
  echo "  \"arch\": \"$(uname -m)\","
  echo "  \"darwin_kernel\": \"$(uname -r)\","
  echo "  \"hf2q_bin\": \"$HF2Q_BIN\","
  echo "  \"fixture\": \"$FIXTURE\","
  echo "  \"prompt\": \"$PROMPT\","
  echo "  \"n_tokens\": $N_TOKENS,"
  echo "  \"n_trials\": $N_TRIALS,"
  echo "  \"thermal_settle_sec\": $THERMAL_SETTLE_SEC,"
  echo "  \"xctrace_version\": \"$($XCTRACE version 2>&1 | head -1)\","
  echo "  \"hf2q_git_head\": \"$(git -C /opt/hf2q rev-parse HEAD)\","
  echo "  \"hf2q_git_dirty\": \"$(git -C /opt/hf2q status --porcelain | tr '\n' ';')\""
  echo "}"
} > "$RUN_META"
echo "wrote run metadata: $RUN_META"

# ----------------------------------------------------------------------------
# Trial loop
# ----------------------------------------------------------------------------
for trial in $(seq 1 "$N_TRIALS"); do
  echo
  echo "--- trial $trial / $N_TRIALS ---"

  # Thermal gate: pmset -g therm shows CPU_Speed_Limit when throttling. On a
  # cold M5 Max with no throttle, the only output is the three "no warning"
  # notes. We detect "CPU_Speed_Limit" as the throttle marker.
  THERM="$(pmset -g therm 2>&1 || true)"
  echo "$THERM"
  if [[ "${SKIP_THERMAL_GATE:-0}" != "1" ]]; then
    if echo "$THERM" | grep -q "CPU_Speed_Limit"; then
      LIMIT="$(echo "$THERM" | awk -F'=' '/CPU_Speed_Limit/ {print $2}' | tr -d ' ')"
      if [[ -n "$LIMIT" && "$LIMIT" != "100" ]]; then
        echo "ERROR: thermal throttle detected (CPU_Speed_Limit=$LIMIT). Aborting." >&2
        exit 2
      fi
    fi
  fi

  TRACE_OUT="$OUT_DIR/trace-${trial}-${DATE_TAG}.trace"
  STDOUT_OUT="$OUT_DIR/trial-${trial}-${DATE_TAG}.stdout"
  STDERR_OUT="$OUT_DIR/trial-${trial}-${DATE_TAG}.stderr"
  META_OUT="$OUT_DIR/trial-${trial}-${DATE_TAG}.metadata.json"

  # Capture per-trial thermal state in JSON.
  {
    echo "{"
    echo "  \"trial\": $trial,"
    echo "  \"date_utc\": \"$(date -u +%Y%m%dT%H%M%SZ)\","
    echo "  \"thermal_pre\": $(printf '%s' "$THERM" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),"
    echo "  \"trace_path\": \"$TRACE_OUT\""
    echo "}"
  } > "$META_OUT"

  # Run the binary under xctrace.
  # Note: --launch passes the binary + args. Use `--` to separate.
  echo "running xctrace → $TRACE_OUT"
  set +e
  "$XCTRACE" record \
    --template "Time Profiler" \
    --output "$TRACE_OUT" \
    --launch -- "$HF2Q_BIN" generate \
      --model "$FIXTURE" \
      --prompt "$PROMPT" \
      --max-tokens "$N_TOKENS" \
      --temperature 0 \
      > "$STDOUT_OUT" 2> "$STDERR_OUT"
  XC_RC=$?
  set -e
  echo "xctrace exit code: $XC_RC"
  if [[ $XC_RC -ne 0 ]]; then
    echo "WARN: xctrace returned non-zero. Check $STDERR_OUT" >&2
  fi

  # Quick top-N export. xctrace's Time Profiler export is XML — we capture
  # both raw export and a digestible summary that the analyst can grep.
  TOPCALLS_OUT="$OUT_DIR/topcalls-${trial}-${DATE_TAG}.txt"
  if [[ -d "$TRACE_OUT" || -f "$TRACE_OUT" ]]; then
    echo "exporting time-profile summary → $TOPCALLS_OUT"
    set +e
    "$XCTRACE" export \
      --input "$TRACE_OUT" \
      --xpath '/trace-toc/run[1]/data/table[@schema="time-profile"]' \
      > "$TOPCALLS_OUT" 2>/dev/null
    EXP_RC=$?
    set -e
    if [[ $EXP_RC -ne 0 || ! -s "$TOPCALLS_OUT" ]]; then
      echo "WARN: xctrace export failed; trace must be opened in Instruments.app for top-call extraction" >&2
    fi
  else
    echo "WARN: trace file not produced, skipping export" >&2
  fi

  # Inter-trial thermal settle (skip after final trial).
  if [[ $trial -lt $N_TRIALS ]]; then
    echo "thermal settle: sleep $THERMAL_SETTLE_SEC s"
    sleep "$THERMAL_SETTLE_SEC"
  fi
done

echo
echo "=== run complete ==="
echo "metadata : $RUN_META"
echo "traces   : $OUT_DIR/trace-*-${DATE_TAG}.trace"
echo "exports  : $OUT_DIR/topcalls-*-${DATE_TAG}.txt"
echo
echo "Next: open one trace in Instruments.app for top-N call site analysis,"
echo "      or grep topcalls-*.txt for hf2q::inference::models::qwen35::"
echo "      function frames. Findings land in ADR-015 §P3a'."
